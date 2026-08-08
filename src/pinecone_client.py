"""Lazy Pinecone client and indexing helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import PineconeError

from src.config import PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_INDEX, PINECONE_REGION
from src.embeddings import EmbeddedChunk


logger = logging.getLogger(__name__)

PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"

_pinecone_client: Pinecone | None = None
_index = None


class PineconeIndexingError(RuntimeError):
    """Raised when Pinecone index operations fail."""


class PineconeQueryResult(Protocol):
    """Protocol for the query response used in verification tests."""

    matches: Sequence[Any]


@dataclass(slots=True)
class PineconeVectorRecord:
    """Vector payload ready for Pinecone upsert."""

    id: str
    values: list[float]
    metadata: dict[str, Any]


class _LazyIndexProxy:
    """Compatibility proxy that resolves the index only when accessed."""

    def __getattr__(self, item: str) -> Any:
        return getattr(get_index(), item)


index = _LazyIndexProxy()


def get_pinecone_client() -> Pinecone:
    """Return the cached Pinecone admin client."""

    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pinecone_client


def ensure_index() -> None:
    """Ensure the configured Pinecone index exists."""

    if not PINECONE_INDEX:
        raise PineconeIndexingError("PINECONE_INDEX is not configured")

    pinecone_client = get_pinecone_client()
    try:
        if pinecone_client.has_index(PINECONE_INDEX):
            logger.debug("Pinecone index %s already exists", PINECONE_INDEX)
            return

        logger.info("Creating Pinecone index %s", PINECONE_INDEX)
        pinecone_client.create_index(
            name=PINECONE_INDEX,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    except PineconeError as exc:
        logger.exception("Failed to ensure Pinecone index %s", PINECONE_INDEX)
        raise PineconeIndexingError(
            f"Failed to ensure Pinecone index {PINECONE_INDEX!r}"
        ) from exc


def get_index():
    """Return the configured Pinecone data-plane index, creating it if needed."""

    global _index
    if _index is not None:
        return _index

    ensure_index()
    pinecone_client = get_pinecone_client()
    try:
        index_model = pinecone_client.describe_index(PINECONE_INDEX)
        if index_model.host is None:
            raise PineconeIndexingError(
                f"Pinecone index {PINECONE_INDEX!r} does not have a data-plane host yet"
            )

        _index = pinecone_client.Index(
            host=index_model.host,
        )
    except PineconeError as exc:  # pragma: no cover - network failures
        raise PineconeIndexingError(
            f"Unable to resolve Pinecone index {PINECONE_INDEX!r}"
        ) from exc
    return _index


def _build_vector_id(chunk: EmbeddedChunk) -> str:
    """Build a stable vector ID from ingestion metadata."""

    metadata = chunk.metadata
    relative_file_path = metadata.get("relative_file_path")
    chunk_id = metadata.get("chunk_id")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise PineconeIndexingError("Chunk metadata is missing a valid chunk_id")
    if not isinstance(relative_file_path, str) or not relative_file_path:
        raise PineconeIndexingError(
            "Chunk metadata is missing a valid relative_file_path"
        )

    if chunk_id == relative_file_path or chunk_id.startswith(f"{relative_file_path}::"):
        # Assumption: ingestion already emitted a globally unique chunk_id.
        return chunk_id

    return f"{relative_file_path}::{chunk_id}"


def _build_vector_record(chunk: EmbeddedChunk) -> PineconeVectorRecord:
    """Convert an embedded chunk into a Pinecone vector record."""

    metadata = dict(chunk.metadata)
    metadata["text"] = chunk.text
    return PineconeVectorRecord(
        id=_build_vector_id(chunk),
        values=list(chunk.embedding),
        metadata=metadata,
    )


def upsert_embedded_chunks(
    chunks: Sequence[EmbeddedChunk],
    namespace: str | None = None,
) -> Any:
    """Upsert embedded chunks into Pinecone."""

    if not chunks:
        logger.info("No embedded chunks supplied for Pinecone upsert")
        return None

    pinecone_index = get_index()
    vectors = [
        (record.id, record.values, record.metadata)
        for record in (_build_vector_record(chunk) for chunk in chunks)
    ]

    logger.info("Upserting %d vectors into Pinecone", len(vectors))
    try:
        return pinecone_index.upsert(
            vectors=vectors,
            namespace=namespace or "",
        )
    except PineconeError as exc:
        logger.exception("Pinecone upsert failed")
        raise PineconeIndexingError("Failed to upsert embedded chunks to Pinecone") from exc


def query_embedding(
    vector: Sequence[float],
    top_k: int = 1,
    namespace: str | None = None,
) -> Any:
    """Query Pinecone with an embedding vector for verification."""

    pinecone_index = get_index()
    try:
        return pinecone_index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=namespace or "",
            include_values=False,
            include_metadata=True,
        )
    except PineconeError as exc:
        logger.exception("Pinecone query failed")
        raise PineconeIndexingError("Failed to query Pinecone") from exc

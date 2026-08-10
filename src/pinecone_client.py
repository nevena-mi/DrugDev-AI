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
PINECONE_UPSERT_BATCH_SIZE = 100

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
    namespace_value = namespace or ""
    total_chunks = len(chunks)
    logger.info(
        "Upserting %d vectors into Pinecone in batches of %d",
        total_chunks,
        PINECONE_UPSERT_BATCH_SIZE,
    )

    last_response: Any = None
    batch: list[tuple[str, list[float], dict[str, Any]]] = []
    batch_start = 0

    for chunk_index, chunk in enumerate(chunks, start=1):
        record = _build_vector_record(chunk)
        batch.append((record.id, record.values, record.metadata))

        if len(batch) < PINECONE_UPSERT_BATCH_SIZE and chunk_index < total_chunks:
            continue

        batch_number = (batch_start // PINECONE_UPSERT_BATCH_SIZE) + 1
        logger.info(
            "Upserting Pinecone batch %d (%d vectors of %d total)",
            batch_number,
            len(batch),
            total_chunks,
        )
        try:
            last_response = pinecone_index.upsert(
                vectors=batch,
                namespace=namespace_value,
            )
        except PineconeError as exc:
            logger.exception("Pinecone upsert failed for batch %d", batch_number)
            raise PineconeIndexingError(
                "Failed to upsert embedded chunks to Pinecone"
            ) from exc

        logger.info("Completed Pinecone batch %d", batch_number)
        batch = []
        batch_start = chunk_index

    logger.info("Finished upserting %d vectors into Pinecone", total_chunks)
    return last_response


def query_embedding(
    vector: Sequence[float],
    top_k: int = 1,
    namespace: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> Any:
    """Query Pinecone with an embedding vector for verification."""

    pinecone_index = get_index()
    query_kwargs: dict[str, Any] = {
        "vector": list(vector),
        "top_k": top_k,
        "namespace": namespace or "",
        "include_values": False,
        "include_metadata": True,
    }
    if metadata_filter is not None:
        query_kwargs["filter"] = metadata_filter
    try:
        return pinecone_index.query(**query_kwargs)
    except PineconeError as exc:
        logger.exception("Pinecone query failed")
        raise PineconeIndexingError("Failed to query Pinecone") from exc

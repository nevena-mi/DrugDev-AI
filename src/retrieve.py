"""Semantic retrieval over Pinecone-indexed document chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from src.embeddings import embed_query
from src.pinecone_client import query_embedding


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A ranked retrieval result returned from Pinecone."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]


def _extract_matches(response: Any) -> Sequence[Any]:
    """Return Pinecone matches from a response-like object."""

    matches = getattr(response, "matches", None)
    if not matches:
        return []
    return matches


def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    namespace: str | None = None,
    document_paths: Sequence[str] | None = None,
    cost_mode: str = "unknown",
) -> list[RetrievedChunk]:
    """Embed a natural-language query and return the most relevant chunks."""

    logger.info("Retrieving chunks for query %r", query)
    vector = embed_query(query, mode=cost_mode)
    metadata_filter = None
    if document_paths is not None:
        filtered_paths = [path for path in document_paths if path]
        if not filtered_paths:
            logger.info("Document-path filter resolved to no usable paths for query %r", query)
            return []
        metadata_filter = {"relative_file_path": {"$in": filtered_paths}}

    response = query_embedding(
        vector,
        top_k=top_k,
        namespace=namespace,
        metadata_filter=metadata_filter,
    )

    matches = _extract_matches(response)
    if not matches:
        logger.info("No retrieval matches found for query %r", query)
        return []

    ranked_matches = sorted(
        matches,
        key=lambda match: float(getattr(match, "score", 0.0) or 0.0),
        reverse=True,
    )

    retrieved_chunks = [
        RetrievedChunk(
            id=str(getattr(match, "id", "")),
            score=float(getattr(match, "score", 0.0) or 0.0),
            text=str((getattr(match, "metadata", {}) or {}).get("text", "")),
            metadata=dict(getattr(match, "metadata", {}) or {}),
        )
        for match in ranked_matches
    ]

    logger.info("Retrieved %d chunks for query %r", len(retrieved_chunks), query)
    return retrieved_chunks

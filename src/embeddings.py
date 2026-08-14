"""OpenAI embedding generation for ingested document chunks.

It batches chunk texts, calls the existing OpenAI client with config.EMBEDDING_MODEL, 
preserves each chunk-s metadata, and returns embedding records ready for the Pinecone phase.
It logs batch start, success, and failure, and wraps API errors in the module-specific exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.config import EMBEDDING_MODEL
from src.openai_client import client
from src.costs import record_openai_embedding


logger = logging.getLogger(__name__)


class EmbeddableDocument(Protocol):
    """Protocol for document chunks produced by ingestion."""

    page_content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class EmbeddedChunk:
    """Embedding payload paired with the original chunk metadata."""

    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class EmbeddingGenerationError(RuntimeError):
    """Raised when embedding generation fails."""


def _embed_texts(
    texts: Sequence[str],
    *,
    phase: str,
    mode: str,
    operation: str,
) -> list[list[float]]:
    """Embed one or more texts using the configured OpenAI model."""

    if not texts:
        return []

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=list(texts),
        )
    except Exception as exc:  # pragma: no cover - exercised via failure test
        logger.exception("Embedding generation failed")
        raise EmbeddingGenerationError("Failed to generate embeddings") from exc

    # Cost analytics must never break the functional embedding workflow.
    try:
        record_openai_embedding(
            response,
            phase=phase,
            mode=mode,
            operation=operation,
            model=EMBEDDING_MODEL,
        )
    except Exception:
        logger.exception("Failed to record embedding cost analytics")

    if len(response.data) != len(texts):  # pragma: no cover - defensive guard
        raise EmbeddingGenerationError("Embedding API returned a mismatched number of vectors")

    return [list(embedding_item.embedding) for embedding_item in response.data]


def _iter_batches(
    documents: Sequence[EmbeddableDocument],
    batch_size: int,
) -> list[Sequence[EmbeddableDocument]]:
    return [
        documents[index : index + batch_size]
        for index in range(0, len(documents), batch_size)
    ]


def generate_embeddings(
    documents: Sequence[EmbeddableDocument],
) -> list[EmbeddedChunk]:
    """Generate embeddings for document chunks using the configured OpenAI model."""

    if not documents:
        logger.info("No document chunks supplied for embedding generation")
        return []

    batch_size = 100
    embedded_chunks: list[EmbeddedChunk] = []

    for batch_number, batch in enumerate(_iter_batches(documents, batch_size), start=1):
        batch_texts = [document.page_content for document in batch]
        logger.info(
            "Generating embeddings for batch %d containing %d chunks with model %s",
            batch_number,
            len(batch),
            EMBEDDING_MODEL,
        )

        embeddings = _embed_texts(batch_texts,
            phase="build",
            mode="ingestion",
            operation="knowledge_base_embedding",)
        for document, embedding in zip(batch, embeddings, strict=True):
            embedded_chunks.append(
                EmbeddedChunk(
                    text=document.page_content,
                    embedding=embedding,
                    metadata=dict(document.metadata),
                )
            )

    logger.info("Generated embeddings for %d document chunks", len(embedded_chunks))
    return embedded_chunks


def embed_query(query: str, *, mode: str = "unknown",) -> list[float]:
    """Generate a single embedding vector for a retrieval query."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise EmbeddingGenerationError("Query text must not be empty")

    embeddings = _embed_texts([cleaned_query], phase="runtime", mode=mode, operation="query_embedding",)
    return embeddings[0]

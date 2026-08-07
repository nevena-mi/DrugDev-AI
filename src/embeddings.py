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

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
            )
        except Exception as exc:  # pragma: no cover - exercised via failure test
            logger.exception("Embedding generation failed for batch %d", batch_number)
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings for batch {batch_number}"
            ) from exc

        if len(response.data) != len(batch):  # pragma: no cover - defensive guard
            raise EmbeddingGenerationError(
                "Embedding API returned a mismatched number of vectors"
            )

        for document, embedding_item in zip(batch, response.data, strict=True):
            embedded_chunks.append(
                EmbeddedChunk(
                    text=document.page_content,
                    embedding=list(embedding_item.embedding),
                    metadata=dict(document.metadata),
                )
            )

    logger.info("Generated embeddings for %d document chunks", len(embedded_chunks))
    return embedded_chunks

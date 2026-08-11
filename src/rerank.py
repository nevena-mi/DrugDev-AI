"""Cohere reranking helpers for evaluation-time ranking experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import cohere

from src.config import COHERE_API_KEY, COHERE_RERANK_MODEL
from src.retrieve import RetrievedChunk


logger = logging.getLogger(__name__)


class CohereRerankClient(Protocol):
    """Protocol for the Cohere client rerank surface used by this module."""

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> Any:
        """Rerank a list of documents for a query."""


@dataclass(slots=True)
class RerankedChunk:
    """A reranked chunk preserving the original retrieval payload."""

    id: str
    text: str
    metadata: dict[str, Any]
    pinecone_score: float
    cohere_score: float
    original_index: int
    reranked_rank: int


class RerankingError(RuntimeError):
    """Raised when Cohere reranking fails or returns invalid output."""


_cohere_client: CohereRerankClient | None = None


def get_cohere_client() -> CohereRerankClient:
    """Return a cached Cohere client configured from project settings."""

    global _cohere_client
    if _cohere_client is None:
        if not COHERE_API_KEY:
            raise RerankingError("COHERE_API_KEY is not configured")
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _cohere_client


def _extract_results(response: Any) -> Sequence[Any]:
    """Return rerank results from a response-like object."""

    results = getattr(response, "results", None)
    if results is None:
        return []
    return results


def rerank_chunks(
    query: str,
    candidates: Sequence[RetrievedChunk],
    *,
    top_n: int | None = None,
    model: str | None = None,
    client: CohereRerankClient | None = None,
) -> list[RerankedChunk]:
    """Rerank candidate chunks using Cohere and preserve the original payload."""

    if not candidates:
        logger.info("No candidate chunks supplied for Cohere reranking")
        return []

    rerank_model = model or COHERE_RERANK_MODEL
    rerank_top_n = min(top_n or len(candidates), len(candidates))
    client = client or get_cohere_client()
    documents = [candidate.text for candidate in candidates]

    logger.info(
        "Reranking %d candidate chunks with model %s and top_n=%d",
        len(candidates),
        rerank_model,
        rerank_top_n,
    )

    try:
        response = client.rerank(
            model=rerank_model,
            query=query,
            documents=documents,
            top_n=rerank_top_n,
        )
    except (
        cohere.UnauthorizedError,
        cohere.ForbiddenError,
        cohere.BadRequestError,
        cohere.UnprocessableEntityError,
        cohere.TooManyRequestsError,
        cohere.InternalServerError,
        cohere.ServiceUnavailableError,
        cohere.GatewayTimeoutError,
    ) as exc:
        logger.exception("Cohere reranking failed")
        raise RerankingError("Failed to rerank candidate chunks with Cohere") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Unexpected Cohere reranking failure")
        raise RerankingError("Failed to rerank candidate chunks with Cohere") from exc

    results = list(_extract_results(response))
    if len(results) != rerank_top_n:
        raise RerankingError(
            "Cohere rerank returned an unexpected number of results"
        )

    reranked_chunks: list[RerankedChunk] = []
    seen_indices: set[int] = set()
    for reranked_rank, result in enumerate(results, start=1):
        try:
            original_index = int(getattr(result, "index"))
            cohere_score = float(getattr(result, "relevance_score"))
        except (TypeError, ValueError) as exc:
            raise RerankingError("Cohere rerank returned invalid result data") from exc

        if original_index < 0 or original_index >= len(candidates):
            raise RerankingError(
                f"Cohere rerank returned an out-of-range index: {original_index}"
            )
        if original_index in seen_indices:
            raise RerankingError(
                f"Cohere rerank returned a duplicate index: {original_index}"
            )

        seen_indices.add(original_index)
        candidate = candidates[original_index]
        reranked_chunks.append(
            RerankedChunk(
                id=candidate.id,
                text=candidate.text,
                metadata=dict(candidate.metadata),
                pinecone_score=float(candidate.score),
                cohere_score=cohere_score,
                original_index=original_index,
                reranked_rank=reranked_rank,
            )
        )

    logger.info(
        "Reranked %d chunks into a Cohere-ranked list of %d items",
        len(candidates),
        len(reranked_chunks),
    )
    return reranked_chunks

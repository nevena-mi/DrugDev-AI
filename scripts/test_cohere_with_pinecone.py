"""Standalone integration script for Cohere reranking on top of Pinecone retrieval."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def _bootstrap_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _print_ranking(
    *,
    heading: str,
    chunks: Sequence,
    scores: Sequence[float] | None = None,
    indices: Sequence[int] | None = None,
) -> None:
    print(heading)
    for position, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("document_title") or chunk.metadata.get("filename") or ""
        original_score = getattr(chunk, "score", None)
        if indices is None:
            print(
                f"{position}. title={title} original_pinecone_score={original_score}"
            )
            continue
        rerank_index = indices[position - 1]
        coh_score = scores[position - 1]
        print(
            f"{position}. original_rank={rerank_index + 1} reranked_rank={position} "
            f"original_pinecone_score={original_score} cohere_relevance_score={coh_score} "
            f"document_title={title}"
        )


def main() -> int:
    """Run the Pinecone + Cohere reranking integration check."""

    _bootstrap_path()
    load_dotenv(ENV_PATH)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import cohere
    except Exception as exc:  # pragma: no cover - environment failure
        print(f"Failed to import Cohere SDK: {type(exc).__name__}: {exc}")
        return 1

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("Missing COHERE_API_KEY in .env")
        return 1

    try:
        from src.retrieve import retrieve_chunks
    except Exception as exc:  # pragma: no cover - environment failure
        print(f"Failed to import project retriever: {type(exc).__name__}: {exc}")
        return 1

    query = "What is pharmacovigilance planning according to ICH E2E?"
    try:
        retrieved_chunks = retrieve_chunks(query, top_k=15)
    except Exception as exc:
        print(f"Failed to retrieve chunks: {type(exc).__name__}: {exc}")
        return 1

    if not retrieved_chunks:
        print("No chunks were retrieved from Pinecone.")
        return 1

    client = cohere.ClientV2(api_key=api_key)
    documents = [chunk.page_content if hasattr(chunk, "page_content") else chunk.text for chunk in retrieved_chunks]

    try:
        response = client.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=documents,
            top_n=min(15, len(documents)),
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
        print(f"Cohere rerank failed: {type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Cohere rerank failed: {type(exc).__name__}: {exc}")
        return 1

    results = list(response.results)
    if len(results) != len(retrieved_chunks):
        print(
            "Cohere rerank returned a different number of results than the retrieved chunks."
        )
        return 1

    reranked_chunks = []
    seen_indices: set[int] = set()
    for result in results:
        index = int(result.index)
        if index < 0 or index >= len(retrieved_chunks):
            print(f"Invalid rerank index returned by Cohere: {index}")
            return 1
        if index in seen_indices:
            print(f"Duplicate rerank index returned by Cohere: {index}")
            return 1
        seen_indices.add(index)
        reranked_chunks.append(retrieved_chunks[index])
    if len(seen_indices) != len(results):
        print("Reranked list does not contain unique original chunks.")
        return 1

    print(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone.")
    _print_ranking(heading="Original ranking:", chunks=retrieved_chunks)
    _print_ranking(
        heading="Reranked ranking:",
        chunks=reranked_chunks,
        scores=[result.relevance_score for result in results],
        indices=[result.index for result in results],
    )

    original_index_set = set(range(len(retrieved_chunks)))
    if seen_indices != original_index_set:
        print("Reranked indices do not cover the full original retrieval set.")
        return 1
    if seen_indices - original_index_set:
        print("Reranked results reference chunks outside the original retrieval set.")
        return 1

    if len(reranked_chunks) != len(results):
        print("Reranked chunk count mismatch.")
        return 1

    print("Integration test passed")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())

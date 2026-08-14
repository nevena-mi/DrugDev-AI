from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import src.graph as graph_module
from src.rerank import RerankedChunk


def _make_chunk(
    *,
    chunk_id: str = "ema/example.pdf::chunk-0",
    title: str = "example",
    text: str = "Good Clinical Practice applies to clinical trials.",
    score: float = 0.91,
) -> graph_module.RetrievedChunk:
    return graph_module.RetrievedChunk(
        id=chunk_id,
        score=score,
        text=text,
        metadata={
            "filename": "example.pdf",
            "relative_file_path": "ema/example.pdf",
            "source_organization": "ema",
            "document_title": title,
            "chunk_id": chunk_id,
            "text": text,
        },
    )


def _make_reranked_chunk(
    chunk: graph_module.RetrievedChunk,
    *,
    cohere_score: float,
    original_index: int,
    reranked_rank: int,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=dict(chunk.metadata),
        pinecone_score=chunk.score,
        cohere_score=cohere_score,
        original_index=original_index,
        reranked_rank=reranked_rank,
    )


def test_retrieval_occurs_before_reranking_and_context_uses_reranked_top_five() -> None:
    order: list[str] = []
    captured_prompt: dict[str, str] = {}
    pinecone_chunks = [_make_chunk(chunk_id=f"chunk-{index}", title=f"Doc {index}", score=0.95 - index * 0.01) for index in range(15)]
    reranked_chunks = [
        _make_reranked_chunk(
            chunk=pinecone_chunks[index],
            cohere_score=0.99 - position * 0.01,
            original_index=index,
            reranked_rank=position + 1,
        )
        for position, index in enumerate([3, 1, 0, 2, 4], start=0)
    ]

    def fake_retrieve_chunks(query: str, *, top_k: int, namespace=None, document_paths=None, cost_mode="unknown",):
        order.append("retrieve")
        assert query == "What is GCP?"
        assert top_k == 15
        assert namespace is None
        assert document_paths is None
        return pinecone_chunks

    def fake_rerank_chunks(query: str, candidates, *, top_n: int, model=None, client=None, cost_mode="unknown",):
        order.append("rerank")
        assert query == "What is GCP?"
        assert candidates == pinecone_chunks
        assert top_n == 5
        return reranked_chunks

    def fake_create(*, model: str, input: str):
        order.append("generate")
        captured_prompt["model"] = model
        captured_prompt["input"] = input
        return SimpleNamespace(output_text="Good Clinical Practice is a quality standard for trials.")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        result = graph_module.run_ask_workflow("What is GCP?")

    assert order == ["retrieve", "rerank", "generate"]
    assert captured_prompt["model"] == "gpt-4o-mini"
    assert "What is GCP?" in captured_prompt["input"]
    assert "Good Clinical Practice applies to clinical trials." in captured_prompt["input"]
    assert "Doc 3" in captured_prompt["input"]
    assert "Doc 1" in captured_prompt["input"]
    assert result.answer == "Good Clinical Practice is a quality standard for trials."
    assert len(result.citations) == 5
    assert [citation.document_title for citation in result.citations] == ["Doc 3", "Doc 1", "Doc 0", "Doc 2", "Doc 4"]
    assert [chunk.id for chunk in result.retrieved_chunks] == ["chunk-3", "chunk-1", "chunk-0", "chunk-2", "chunk-4"]
    assert [chunk.score for chunk in result.retrieved_chunks] == [
        pinecone_chunks[3].score,
        pinecone_chunks[1].score,
        pinecone_chunks[0].score,
        pinecone_chunks[2].score,
        pinecone_chunks[4].score,
    ]


def test_insufficient_retrieval_returns_safe_response_without_llm_call() -> None:
    with (
        patch.object(graph_module, "retrieve_chunks", return_value=[]),
        patch.object(graph_module, "rerank_chunks") as rerank,
        patch.object(graph_module.client.responses, "create") as create,
    ):
        result = graph_module.run_ask_workflow("Unknown topic")

    rerank.assert_not_called()
    create.assert_not_called()
    assert result.answer == graph_module.INSUFFICIENT_INFORMATION
    assert result.citations == []
    assert result.retrieved_chunks == []


def test_reranking_failure_falls_back_to_pinecone_top_five() -> None:
    pinecone_chunks = [_make_chunk(chunk_id=f"chunk-{index}", title=f"Doc {index}", score=0.95 - index * 0.01) for index in range(15)]
    order: list[str] = []
    captured_prompt: dict[str, str] = {}

    def fake_retrieve_chunks(query: str, *, top_k: int, namespace=None, document_paths=None, cost_mode="unknown",):
        order.append("retrieve")
        assert top_k == 15
        return pinecone_chunks

    def fake_rerank_chunks(*args, **kwargs):
        order.append("rerank")
        raise graph_module.RerankingError("boom")

    def fake_create(*, model: str, input: str):
        order.append("generate")
        captured_prompt["input"] = input
        return SimpleNamespace(output_text="Grounded answer")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        result = graph_module.run_ask_workflow("What is GCP?")

    assert order == ["retrieve", "rerank", "generate"]
    assert "Doc 0" in captured_prompt["input"]
    assert "Doc 4" in captured_prompt["input"]
    assert [citation.document_title for citation in result.citations] == ["Doc 0", "Doc 1", "Doc 2", "Doc 3", "Doc 4"]
    assert [chunk.id for chunk in result.retrieved_chunks] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]


def test_llm_failures_are_wrapped_clearly() -> None:
    pinecone_chunks = [_make_chunk()]

    def fake_retrieve_chunks(query: str, *, top_k: int, namespace=None, document_paths=None, cost_mode="unknown"):
        return pinecone_chunks

    def fake_rerank_chunks(*args, **kwargs):
        return [
            _make_reranked_chunk(
                pinecone_chunks[0],
                cohere_score=0.99,
                original_index=0,
                reranked_rank=1,
            )
        ]

    def fake_create(*, model: str, input: str):
        raise RuntimeError("boom")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        try:
            graph_module.run_ask_workflow("What is GCP?")
        except graph_module.RAGGenerationError as exc:
            assert "Failed to generate a grounded answer" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("RAGGenerationError was not raised")

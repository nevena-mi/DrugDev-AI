from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import src.graph as graph_module


def _make_chunk() -> graph_module.RetrievedChunk:
    return graph_module.RetrievedChunk(
        id="ema/example.pdf::chunk-0",
        score=0.91,
        text="Good Clinical Practice applies to clinical trials.",
        metadata={
            "filename": "example.pdf",
            "relative_file_path": "ema/example.pdf",
            "source_organization": "ema",
            "document_title": "example",
            "chunk_id": "ema/example.pdf::chunk-0",
            "text": "Good Clinical Practice applies to clinical trials.",
        },
    )


def test_retrieval_occurs_before_generation_and_context_is_passed_to_llm() -> None:
    order: list[str] = []
    captured_prompt: dict[str, str] = {}

    def fake_retrieve_chunks(query: str, *, top_k: int, namespace=None):
        order.append("retrieve")
        assert query == "What is GCP?"
        assert top_k == 2
        assert namespace is None
        return [_make_chunk()]

    def fake_create(*, model: str, input: str):
        order.append("generate")
        captured_prompt["model"] = model
        captured_prompt["input"] = input
        return SimpleNamespace(output_text="Good Clinical Practice is a quality standard for trials.")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        result = graph_module.run_ask_workflow("What is GCP?", top_k=2)

    assert order == ["retrieve", "generate"]
    assert captured_prompt["model"] == "gpt-4o-mini"
    assert "What is GCP?" in captured_prompt["input"]
    assert "Good Clinical Practice applies to clinical trials." in captured_prompt["input"]
    assert result.answer == "Good Clinical Practice is a quality standard for trials."
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.filename == "example.pdf"
    assert citation.relative_file_path == "ema/example.pdf"
    assert citation.document_title == "example"
    assert result.retrieved_chunks[0].id == "ema/example.pdf::chunk-0"


def test_insufficient_retrieval_returns_safe_response_without_llm_call() -> None:
    with (
        patch.object(graph_module, "retrieve_chunks", return_value=[]),
        patch.object(graph_module.client.responses, "create") as create,
    ):
        result = graph_module.run_ask_workflow("Unknown topic")

    create.assert_not_called()
    assert result.answer == graph_module.INSUFFICIENT_INFORMATION
    assert result.citations == []
    assert result.retrieved_chunks == []


def test_llm_failures_are_wrapped_clearly() -> None:
    def fake_retrieve_chunks(query: str, *, top_k: int, namespace=None):
        return [_make_chunk()]

    def fake_create(*, model: str, input: str):
        raise RuntimeError("boom")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        try:
            graph_module.run_ask_workflow("What is GCP?")
        except graph_module.RAGGenerationError as exc:
            assert "Failed to generate a grounded answer" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("RAGGenerationError was not raised")


from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.embeddings import EmbeddedChunk, EmbeddingGenerationError, generate_embeddings
from src.ingest import Document


def _make_document(index: int) -> Document:
    return Document(
        page_content=f"Chunk text {index}",
        metadata={
            "filename": f"document-{index}.pdf",
            "relative_file_path": f"org/document-{index}.pdf",
            "source_organization": "org",
            "document_title": f"document-{index}",
            "chunk_id": f"org/document-{index}.pdf::chunk-0",
        },
    )


def test_generate_embeddings_returns_embeddings_and_metadata() -> None:
    documents = [_make_document(1), _make_document(2)]
    calls: list[dict[str, object]] = []

    def fake_create(*, model: str, input: list[str]) -> SimpleNamespace:
        calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )

    with patch("src.embeddings.client.embeddings.create", fake_create):
        result = generate_embeddings(documents)

    assert len(result) == 2
    assert all(isinstance(chunk, EmbeddedChunk) for chunk in result)
    assert [chunk.text for chunk in result] == [document.page_content for document in documents]
    assert [chunk.embedding for chunk in result] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert [chunk.metadata for chunk in result] == [document.metadata for document in documents]
    assert all(len(chunk.embedding) == 3 for chunk in result)
    assert calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["Chunk text 1", "Chunk text 2"],
        }
    ]


def test_generate_embeddings_wraps_api_failures() -> None:
    def fake_create(*, model: str, input: list[str]) -> SimpleNamespace:
        raise RuntimeError("boom")

    try:
        with patch("src.embeddings.client.embeddings.create", fake_create):
            generate_embeddings([_make_document(1)])
    except EmbeddingGenerationError as exc:
        assert "Failed to generate embeddings" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("EmbeddingGenerationError was not raised")


def test_generate_embeddings_returns_empty_list_for_no_documents() -> None:
    assert generate_embeddings([]) == []

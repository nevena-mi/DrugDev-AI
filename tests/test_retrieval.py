from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.embeddings import embed_query
import src.retrieve as retrieve_module


def test_embed_query_generates_single_embedding_vector() -> None:
    fake_response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.11, 0.22, 0.33])]
    )

    with patch("src.embeddings.client.embeddings.create", return_value=fake_response) as create:
        vector = embed_query("What is GCP?")

    assert vector == [0.11, 0.22, 0.33]
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == ["What is GCP?"]


def test_retrieve_chunks_uses_query_embedding_and_returns_ranked_matches() -> None:
    fake_matches = [
        SimpleNamespace(
            id="chunk-low",
            score=0.42,
            metadata={"filename": "low.pdf", "chunk_id": "low-0"},
        ),
        SimpleNamespace(
            id="chunk-high",
            score=0.91,
            metadata={"filename": "high.pdf", "chunk_id": "high-0"},
        ),
    ]
    fake_response = SimpleNamespace(matches=fake_matches)

    with (
        patch.object(retrieve_module, "embed_query", return_value=[0.1, 0.2, 0.3]) as embed,
        patch.object(retrieve_module, "query_embedding", return_value=fake_response) as query,
    ):
        results = retrieve_module.retrieve_chunks("Find the guidance", top_k=2)

    embed.assert_called_once_with("Find the guidance")
    query.assert_called_once_with([0.1, 0.2, 0.3], top_k=2, namespace=None, metadata_filter=None)
    assert [result.id for result in results] == ["chunk-high", "chunk-low"]
    assert [result.score for result in results] == [0.91, 0.42]
    assert results[0].metadata == {"filename": "high.pdf", "chunk_id": "high-0"}
    assert results[1].metadata == {"filename": "low.pdf", "chunk_id": "low-0"}


def test_retrieve_chunks_returns_empty_list_for_empty_matches() -> None:
    fake_response = SimpleNamespace(matches=[])

    with (
        patch.object(retrieve_module, "embed_query", return_value=[0.1, 0.2, 0.3]),
        patch.object(retrieve_module, "query_embedding", return_value=fake_response),
    ):
        results = retrieve_module.retrieve_chunks("Empty result query")

    assert results == []


def test_retrieve_chunks_applies_document_path_filter() -> None:
    fake_response = SimpleNamespace(
        matches=[
            SimpleNamespace(
                id="chunk-high",
                score=0.91,
                metadata={"filename": "high.pdf", "chunk_id": "high-0"},
            )
        ]
    )

    with (
        patch.object(retrieve_module, "embed_query", return_value=[0.1, 0.2, 0.3]),
        patch.object(retrieve_module, "query_embedding", return_value=fake_response) as query,
    ):
        results = retrieve_module.retrieve_chunks(
            "Find the guidance",
            top_k=2,
            document_paths=["ich/ich_e6_r3.pdf", "wma/declaration_of_helsinki.pdf"],
        )

    query.assert_called_once_with(
        [0.1, 0.2, 0.3],
        top_k=2,
        namespace=None,
        metadata_filter={
            "relative_file_path": {
                "$in": ["ich/ich_e6_r3.pdf", "wma/declaration_of_helsinki.pdf"]
            }
        },
    )
    assert [result.id for result in results] == ["chunk-high"]

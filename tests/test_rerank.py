from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rerank import RerankedChunk, RerankingError, rerank_chunks
from src.retrieve import RetrievedChunk


def test_rerank_chunks_maps_indices_and_preserves_payload() -> None:
    candidates = [
        RetrievedChunk(
            id="chunk-0",
            score=0.11,
            text="text 0",
            metadata={"document_title": "Doc 0", "topic": "A"},
        ),
        RetrievedChunk(
            id="chunk-1",
            score=0.22,
            text="text 1",
            metadata={"document_title": "Doc 1", "topic": "B"},
        ),
        RetrievedChunk(
            id="chunk-2",
            score=0.33,
            text="text 2",
            metadata={"document_title": "Doc 2", "topic": "C"},
        ),
    ]
    fake_response = SimpleNamespace(
        results=[
            SimpleNamespace(index=2, relevance_score=0.91),
            SimpleNamespace(index=0, relevance_score=0.82),
        ]
    )
    fake_client = Mock()
    fake_client.rerank.return_value = fake_response

    reranked = rerank_chunks(
        "What is the guidance?",
        candidates,
        top_n=2,
        model="rerank-v4.0-pro",
        client=fake_client,
    )

    fake_client.rerank.assert_called_once_with(
        model="rerank-v4.0-pro",
        query="What is the guidance?",
        documents=["text 0", "text 1", "text 2"],
        top_n=2,
    )
    assert [chunk.id for chunk in reranked] == ["chunk-2", "chunk-0"]
    assert [chunk.text for chunk in reranked] == ["text 2", "text 0"]
    assert [chunk.metadata for chunk in reranked] == [
        {"document_title": "Doc 2", "topic": "C"},
        {"document_title": "Doc 0", "topic": "A"},
    ]
    assert [chunk.pinecone_score for chunk in reranked] == [0.33, 0.11]
    assert [chunk.cohere_score for chunk in reranked] == [0.91, 0.82]
    assert [chunk.original_index for chunk in reranked] == [2, 0]
    assert [chunk.reranked_rank for chunk in reranked] == [1, 2]
    assert candidates[0].metadata == {"document_title": "Doc 0", "topic": "A"}


def test_rerank_chunks_rejects_duplicate_indices() -> None:
    candidates = [
        RetrievedChunk(id="chunk-0", score=0.11, text="text 0", metadata={}),
        RetrievedChunk(id="chunk-1", score=0.22, text="text 1", metadata={}),
    ]
    fake_response = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.91),
            SimpleNamespace(index=1, relevance_score=0.82),
        ]
    )
    fake_client = Mock()
    fake_client.rerank.return_value = fake_response

    with pytest.raises(RerankingError):
        rerank_chunks("query", candidates, top_n=2, client=fake_client)


def test_rerank_chunks_wraps_api_failures() -> None:
    candidates = [
        RetrievedChunk(id="chunk-0", score=0.11, text="text 0", metadata={}),
        RetrievedChunk(id="chunk-1", score=0.22, text="text 1", metadata={}),
    ]
    fake_client = Mock()
    fake_client.rerank.side_effect = RuntimeError("boom")

    with pytest.raises(RerankingError):
        rerank_chunks("query", candidates, top_n=2, client=fake_client)

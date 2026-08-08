from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

try:  # pragma: no cover - optional in local workspace
    import pytest
except ImportError:  # pragma: no cover - local fallback
    pytest = None

from pinecone.errors.exceptions import PineconeError

from src.embeddings import EmbeddedChunk
import src.pinecone_client as pinecone_client
from src.config import PINECONE_API_KEY, PINECONE_INDEX

TEST_INDEX_NAME = "test-index"


def _make_chunk(
    *,
    relative_file_path: str = "ema/example.pdf",
    chunk_id: str = "chunk-0",
) -> EmbeddedChunk:
    source_organization = relative_file_path.split("/", 1)[0]
    return EmbeddedChunk(
        text="example chunk",
        embedding=[0.1, 0.2, 0.3],
        metadata={
            "filename": "example.pdf",
            "relative_file_path": relative_file_path,
            "source_organization": source_organization,
            "document_title": "example",
            "chunk_id": chunk_id,
        },
    )


def _integration_enabled() -> bool:
    return os.getenv("RUN_PINECONE_INTEGRATION_TESTS") == "1"


def test_module_import_is_lazy() -> None:
    with patch("pinecone.Pinecone") as pinecone_cls:
        reloaded = importlib.reload(pinecone_client)

    assert pinecone_cls.call_count == 0
    assert reloaded._pinecone_client is None
    assert reloaded._index is None


def test_ensure_index_creates_index_when_missing() -> None:
    fake_client = Mock()
    fake_client.has_index.return_value = False

    with (
        patch.object(pinecone_client, "PINECONE_INDEX", TEST_INDEX_NAME),
        patch.object(pinecone_client, "get_pinecone_client", return_value=fake_client),
    ):
        pinecone_client.ensure_index()

    fake_client.has_index.assert_called_once_with(TEST_INDEX_NAME)
    fake_client.create_index.assert_called_once()
    kwargs = fake_client.create_index.call_args.kwargs
    assert kwargs["name"] == TEST_INDEX_NAME
    assert kwargs["dimension"] == pinecone_client.PINECONE_DIMENSION
    assert kwargs["metric"] == pinecone_client.PINECONE_METRIC


def test_ensure_index_skips_creation_when_present() -> None:
    fake_client = Mock()
    fake_client.has_index.return_value = True

    with (
        patch.object(pinecone_client, "PINECONE_INDEX", TEST_INDEX_NAME),
        patch.object(pinecone_client, "get_pinecone_client", return_value=fake_client),
    ):
        pinecone_client.ensure_index()

    fake_client.has_index.assert_called_once_with(TEST_INDEX_NAME)
    fake_client.create_index.assert_not_called()


def test_get_index_resolves_lazily() -> None:
    fake_index = Mock()
    fake_client = Mock()
    fake_client.has_index.return_value = True
    fake_client.describe_index.return_value = SimpleNamespace(host="https://example-host")
    fake_client.Index.return_value = fake_index

    with (
        patch.object(pinecone_client, "_index", None),
        patch.object(pinecone_client, "PINECONE_INDEX", TEST_INDEX_NAME),
        patch.object(pinecone_client, "PINECONE_API_KEY", "test-key"),
        patch.object(pinecone_client, "get_pinecone_client", return_value=fake_client),
    ):
        result = pinecone_client.get_index()

    assert result is fake_index
    fake_client.has_index.assert_called_once_with(TEST_INDEX_NAME)
    fake_client.describe_index.assert_called_once_with(TEST_INDEX_NAME)
    fake_client.Index.assert_called_once_with(host="https://example-host")


def test_upsert_embedded_chunks_builds_stable_ids_and_preserves_metadata() -> None:
    fake_index = Mock()
    fake_index.upsert.return_value = SimpleNamespace(upserted_count=2)
    chunks = [
        _make_chunk(),
        _make_chunk(
            relative_file_path="fda/example.pdf",
            chunk_id="fda/example.pdf::chunk-1",
        ),
    ]

    with patch.object(pinecone_client, "get_index", return_value=fake_index):
        response = pinecone_client.upsert_embedded_chunks(chunks, namespace="phase4")

    assert response is fake_index.upsert.return_value
    fake_index.upsert.assert_called_once()
    kwargs = fake_index.upsert.call_args.kwargs
    assert kwargs["namespace"] == "phase4"
    assert kwargs["vectors"] == [
        (
            "ema/example.pdf::chunk-0",
            [0.1, 0.2, 0.3],
            {
                "filename": "example.pdf",
                "relative_file_path": "ema/example.pdf",
                "source_organization": "ema",
                "document_title": "example",
                "chunk_id": "chunk-0",
                "text": "example chunk",
            },
        ),
        (
            "fda/example.pdf::chunk-1",
            [0.1, 0.2, 0.3],
            {
                "filename": "example.pdf",
                "relative_file_path": "fda/example.pdf",
                "source_organization": "fda",
                "document_title": "example",
                "chunk_id": "fda/example.pdf::chunk-1",
                "text": "example chunk",
            },
        ),
    ]


def test_upsert_embedded_chunks_batches_large_inputs() -> None:
    fake_index = Mock()
    responses = [
        SimpleNamespace(upserted_count=100),
        SimpleNamespace(upserted_count=100),
        SimpleNamespace(upserted_count=5),
    ]
    fake_index.upsert.side_effect = responses
    chunks = [
        _make_chunk(
            relative_file_path="ema/example.pdf",
            chunk_id=f"chunk-{index}",
        )
        for index in range(205)
    ]

    with patch.object(pinecone_client, "get_index", return_value=fake_index):
        response = pinecone_client.upsert_embedded_chunks(chunks, namespace="phase4")

    assert response is responses[-1]
    assert fake_index.upsert.call_count == 3

    first_call = fake_index.upsert.call_args_list[0].kwargs
    second_call = fake_index.upsert.call_args_list[1].kwargs
    third_call = fake_index.upsert.call_args_list[2].kwargs

    assert first_call["namespace"] == "phase4"
    assert second_call["namespace"] == "phase4"
    assert third_call["namespace"] == "phase4"

    assert len(first_call["vectors"]) == pinecone_client.PINECONE_UPSERT_BATCH_SIZE
    assert len(second_call["vectors"]) == pinecone_client.PINECONE_UPSERT_BATCH_SIZE
    assert len(third_call["vectors"]) == 5

    assert first_call["vectors"][0][0] == "ema/example.pdf::chunk-0"
    assert first_call["vectors"][-1][0] == "ema/example.pdf::chunk-99"
    assert second_call["vectors"][0][0] == "ema/example.pdf::chunk-100"
    assert second_call["vectors"][-1][0] == "ema/example.pdf::chunk-199"
    assert third_call["vectors"][0][0] == "ema/example.pdf::chunk-200"
    assert first_call["vectors"][0][2]["chunk_id"] == "chunk-0"
    assert second_call["vectors"][0][2]["chunk_id"] == "chunk-100"
    assert third_call["vectors"][0][2]["chunk_id"] == "chunk-200"


def test_query_embedding_uses_metadata() -> None:
    fake_index = Mock()
    fake_index.query.return_value = SimpleNamespace(
        matches=[SimpleNamespace(id="ema/example.pdf::chunk-0", metadata={"chunk_id": "chunk-0"})]
    )

    with patch.object(pinecone_client, "get_index", return_value=fake_index):
        response = pinecone_client.query_embedding([0.1, 0.2, 0.3], top_k=3, namespace="phase4")

    assert response is fake_index.query.return_value
    fake_index.query.assert_called_once()
    kwargs = fake_index.query.call_args.kwargs
    assert kwargs["vector"] == [0.1, 0.2, 0.3]
    assert kwargs["top_k"] == 3
    assert kwargs["namespace"] == "phase4"
    assert kwargs["include_values"] is False
    assert kwargs["include_metadata"] is True


def test_upsert_and_query_wrap_pinecone_errors() -> None:
    fake_index = Mock()
    fake_index.upsert.side_effect = PineconeError("boom")
    fake_index.query.side_effect = PineconeError("boom")

    with patch.object(pinecone_client, "get_index", return_value=fake_index):
        try:
            pinecone_client.upsert_embedded_chunks([_make_chunk()])
        except pinecone_client.PineconeIndexingError as exc:
            assert "Failed to upsert embedded chunks" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("PineconeIndexingError was not raised for upsert")

    with patch.object(pinecone_client, "get_index", return_value=fake_index):
        try:
            pinecone_client.query_embedding([0.1, 0.2, 0.3])
        except pinecone_client.PineconeIndexingError as exc:
            assert "Failed to query Pinecone" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("PineconeIndexingError was not raised for query")


def test_pinecone_round_trip_integration() -> None:
    if not _integration_enabled():
        if pytest is not None:
            pytest.skip("Set RUN_PINECONE_INTEGRATION_TESTS=1 to enable this test")
        return

    if not PINECONE_API_KEY or not PINECONE_INDEX:
        if pytest is not None:
            pytest.skip("PINECONE_API_KEY and PINECONE_INDEX are required for integration")
        return

    namespace = f"phase4-{uuid4().hex}"
    vector = [0.0] * pinecone_client.PINECONE_DIMENSION
    vector[0] = 1.0
    chunk = EmbeddedChunk(
        text="integration chunk",
        embedding=vector,
        metadata={
            "filename": "integration.pdf",
            "relative_file_path": "integration/integration.pdf",
            "source_organization": "integration",
            "document_title": "integration",
            "chunk_id": "chunk-0",
        },
    )

    upsert_response = pinecone_client.upsert_embedded_chunks([chunk], namespace=namespace)
    query_response = pinecone_client.query_embedding(vector, namespace=namespace)

    assert upsert_response is not None
    assert query_response.matches
    match = query_response.matches[0]
    assert match.id == "integration/integration.pdf::chunk-0"
    assert match.metadata["chunk_id"] == "chunk-0"
    assert match.metadata["relative_file_path"] == "integration/integration.pdf"

from __future__ import annotations

from unittest.mock import patch

import src.chatbot as chatbot_module
import src.graph as graph_module


def test_ask_question_delegates_to_workflow() -> None:
    expected = graph_module.RAGResult(
        question="What is GCP?",
        answer="Grounded answer",
        citations=[],
        retrieved_chunks=[],
    )

    with patch.object(chatbot_module, "run_ask_workflow", return_value=expected) as run:
        result = chatbot_module.ask_question("What is GCP?", top_k=3, namespace="ask")

    run.assert_called_once_with("What is GCP?", top_k=3, namespace="ask")
    assert result is expected


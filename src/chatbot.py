"""Ask-mode entrypoint for the RAG workflow."""

from __future__ import annotations

from src.graph import RAGResult, run_ask_workflow


def ask_question(
    question: str,
    *,
    top_k: int = 5,
    namespace: str | None = None,
) -> RAGResult:
    """Answer a user question with grounded retrieval-augmented generation."""

    return run_ask_workflow(question, top_k=top_k, namespace=namespace)


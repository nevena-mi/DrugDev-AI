"""Ask-mode RAG workflow for grounded question answering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import CHAT_MODEL
from src.openai_client import client
from src.retrieve import RetrievedChunk, retrieve_chunks


logger = logging.getLogger(__name__)

INSUFFICIENT_INFORMATION = "I cannot answer from the available documents."
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "answer.txt"


class RAGGenerationError(RuntimeError):
    """Raised when the answer-generation step fails."""


@dataclass(slots=True)
class SourceCitation:
    """Citation metadata for a retrieved chunk."""

    id: str
    filename: str | None
    relative_file_path: str | None
    source_organization: str | None
    document_title: str | None
    chunk_id: str | None
    score: float

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "SourceCitation":
        """Build a citation from a retrieved chunk."""

        metadata = chunk.metadata
        return cls(
            id=chunk.id,
            filename=metadata.get("filename"),
            relative_file_path=metadata.get("relative_file_path"),
            source_organization=metadata.get("source_organization"),
            document_title=metadata.get("document_title"),
            chunk_id=metadata.get("chunk_id"),
            score=chunk.score,
        )


@dataclass(slots=True)
class AskState:
    """Minimal workflow state for Ask-mode RAG."""

    question: str
    top_k: int = 5
    namespace: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    llm_output: str = ""
    answer: str = ""
    citations: list[SourceCitation] = field(default_factory=list)


@dataclass(slots=True)
class RAGResult:
    """Structured response returned by the Ask workflow."""

    question: str
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]


def _load_prompt_template() -> str:
    """Load the answer-generation prompt from disk."""

    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RAGGenerationError(f"Prompt file not found: {PROMPT_PATH}") from exc


@lru_cache(maxsize=1)
def _get_prompt_template() -> str:
    return _load_prompt_template()


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single prompt context."""

    if not chunks:
        return ""

    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        sections.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"File: {metadata.get('relative_file_path', '')}",
                    f"Document title: {metadata.get('document_title', '')}",
                    f"Chunk ID: {metadata.get('chunk_id', '')}",
                    f"Score: {chunk.score:.4f}",
                    "Text:",
                    chunk.text.strip(),
                ]
            ).strip()
        )
    return "\n\n---\n\n".join(sections)


def _retrieve(state: AskState) -> AskState:
    """Retrieve relevant chunks for the user question."""

    logger.info("Retrieving context for question %r", state.question)
    state.retrieved_chunks = retrieve_chunks(
        state.question,
        top_k=state.top_k,
        namespace=state.namespace,
    )
    state.context = _build_context(state.retrieved_chunks)
    return state


def _generate(state: AskState) -> AskState:
    """Generate a grounded answer from retrieved context."""

    if not state.retrieved_chunks or not state.context.strip():
        logger.info("No usable context retrieved for question %r", state.question)
        state.answer = INSUFFICIENT_INFORMATION
        return state

    prompt_template = _get_prompt_template()
    prompt = prompt_template.format(question=state.question.strip(), context=state.context)

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
        )
    except Exception as exc:  # pragma: no cover - exercised via failure test
        logger.exception("Answer generation failed")
        raise RAGGenerationError("Failed to generate a grounded answer") from exc

    output_text = getattr(response, "output_text", "")
    generated_answer = output_text.strip()
    if not generated_answer:
        raise RAGGenerationError("The model returned an empty answer")

    if generated_answer == INSUFFICIENT_INFORMATION:
        state.answer = INSUFFICIENT_INFORMATION
    else:
        state.answer = generated_answer

    state.llm_output = generated_answer
    return state


def _respond(state: AskState) -> AskState:
    """Attach citations and finalize the response."""

    state.citations = [SourceCitation.from_chunk(chunk) for chunk in state.retrieved_chunks]
    if not state.answer:
        state.answer = INSUFFICIENT_INFORMATION
    return state


class AskRAGWorkflow:
    """Minimal three-node Ask-mode workflow."""

    def invoke(
        self,
        question: str,
        *,
        top_k: int = 5,
        namespace: str | None = None,
    ) -> RAGResult:
        """Run retrieve, generate, and respond for a single question."""

        state = AskState(question=question, top_k=top_k, namespace=namespace)
        state = _retrieve(state)
        state = _generate(state)
        state = _respond(state)
        return RAGResult(
            question=state.question,
            answer=state.answer,
            citations=state.citations,
            retrieved_chunks=state.retrieved_chunks,
        )


_workflow = AskRAGWorkflow()


def run_ask_workflow(
    question: str,
    *,
    top_k: int = 5,
    namespace: str | None = None,
) -> RAGResult:
    """Public helper that executes the Ask-mode RAG workflow."""

    return _workflow.invoke(question, top_k=top_k, namespace=namespace)


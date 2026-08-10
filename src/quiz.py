"""Quiz generation and evaluation for Learn mode."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.config import CHAT_MODEL
from src.curriculum import CurriculumModule
from src.openai_client import client
from src.retrieve import RetrievedChunk


logger = logging.getLogger(__name__)

QUIZ_PASS_THRESHOLD = 0.67


class QuizGenerationError(RuntimeError):
    """Raised when quiz generation fails."""


class QuizEvaluationError(RuntimeError):
    """Raised when quiz evaluation fails."""


@dataclass(slots=True)
class QuizQuestion:
    """A single curriculum quiz question."""

    id: str
    question: str
    reference_answer: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedQuiz:
    """Structured quiz content for a curriculum module."""

    module_id: str
    module_title: str
    questions: list[QuizQuestion]
    source_document_titles: list[str]
    context_summary: str


@dataclass(slots=True)
class QuizEvaluation:
    """Structured quiz evaluation result."""

    module_id: str
    score: float
    passed: bool
    feedback: str
    question_feedback: list[str]


def _unique_document_titles(retrieved_chunks: Sequence[RetrievedChunk]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        title = str(chunk.metadata.get("document_title", "")).strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _build_context(retrieved_chunks: Sequence[RetrievedChunk]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
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


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty response")

    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        loaded = json.loads(match.group(0))

    if not isinstance(loaded, dict):
        raise ValueError("response must be a JSON object")
    return loaded


def _as_question(item: Any, fallback_id: str) -> QuizQuestion:
    if isinstance(item, dict):
        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        source_chunk_ids = item.get("source_chunk_ids", [])
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []
        return QuizQuestion(
            id=str(item.get("id", fallback_id)).strip() or fallback_id,
            question=question,
            reference_answer=reference_answer,
            source_chunk_ids=[str(chunk_id) for chunk_id in source_chunk_ids if str(chunk_id).strip()],
        )
    raise QuizGenerationError("Quiz questions must be returned as objects")


def generate_quiz(
    module: CurriculumModule,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    question_count: int = 3,
) -> GeneratedQuiz:
    """Generate a short quiz from retrieved curriculum context."""

    if not retrieved_chunks:
        raise QuizGenerationError("Cannot generate a quiz without retrieved curriculum context")

    context = _build_context(retrieved_chunks)
    prompt = "\n".join(
        [
            f"You are generating a short knowledge check for the curriculum module '{module.title}'.",
            f"Module id: {module.id}",
            "Use only the provided context.",
            f"Create {question_count} concise questions with reference answers.",
            "Return strict JSON with this shape:",
            '{"questions": [{"id": "q1", "question": "...", "reference_answer": "...", "source_chunk_ids": []}]}',
            "Context:",
            context,
        ]
    )

    try:
        response = client.responses.create(model=CHAT_MODEL, input=prompt)
    except Exception as exc:  # pragma: no cover - exercised via failure tests
        logger.exception("Quiz generation failed")
        raise QuizGenerationError("Failed to generate a curriculum quiz") from exc

    output_text = getattr(response, "output_text", "")
    try:
        payload = _extract_json_object(output_text)
    except Exception as exc:  # pragma: no cover - guarded by tests
        raise QuizGenerationError("Quiz generation must return JSON") from exc

    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list) or not raw_questions:
        raise QuizGenerationError("Quiz generation returned no questions")

    questions = [_as_question(item, f"q{index}") for index, item in enumerate(raw_questions, start=1)]
    if len(questions) < 2:
        raise QuizGenerationError("Quiz generation must return at least two questions")
    questions = questions[:question_count]
    source_document_titles = _unique_document_titles(retrieved_chunks)

    logger.info("Generated %d quiz questions for module %s", len(questions), module.id)
    return GeneratedQuiz(
        module_id=module.id,
        module_title=module.title,
        questions=questions,
        source_document_titles=source_document_titles,
        context_summary=context,
    )


def evaluate_quiz(
    quiz: GeneratedQuiz,
    learner_answers: Sequence[str],
    retrieved_chunks: Sequence[RetrievedChunk],
) -> QuizEvaluation:
    """Evaluate quiz answers using only retrieved curriculum context."""

    if len(learner_answers) != len(quiz.questions):
        raise QuizEvaluationError("Answer count must match the quiz question count")

    context = _build_context(retrieved_chunks)
    answers_payload = [
        {
            "id": question.id,
            "question": question.question,
            "reference_answer": question.reference_answer,
            "learner_answer": answer.strip(),
        }
        for question, answer in zip(quiz.questions, learner_answers, strict=True)
    ]

    prompt = "\n".join(
        [
            f"Evaluate this quiz for module '{quiz.module_title}' ({quiz.module_id}).",
            "Use only the provided context and reference answers.",
            "Return strict JSON with fields: score, passed, feedback, question_feedback.",
            "Question/answer pairs:",
            json.dumps(answers_payload, ensure_ascii=False),
            "Context:",
            context,
        ]
    )

    try:
        response = client.responses.create(model=CHAT_MODEL, input=prompt)
    except Exception as exc:  # pragma: no cover - exercised via failure tests
        logger.exception("Quiz evaluation failed")
        raise QuizEvaluationError("Failed to evaluate the quiz") from exc

    output_text = getattr(response, "output_text", "")
    try:
        payload = _extract_json_object(output_text)
    except Exception as exc:  # pragma: no cover - guarded by tests
        raise QuizEvaluationError("Quiz evaluation must return JSON") from exc

    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError) as exc:
        raise QuizEvaluationError("Quiz evaluation score must be numeric") from exc

    question_feedback = payload.get("question_feedback", [])
    if not isinstance(question_feedback, list):
        question_feedback = []

    passed = score >= QUIZ_PASS_THRESHOLD
    feedback = str(payload.get("feedback", "")).strip()

    logger.info(
        "Evaluated quiz for module %s with score %.2f and passed=%s",
        quiz.module_id,
        score,
        passed,
    )
    return QuizEvaluation(
        module_id=quiz.module_id,
        score=score,
        passed=passed,
        feedback=feedback,
        question_feedback=[str(item) for item in question_feedback],
    )

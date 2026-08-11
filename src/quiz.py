"""Quiz generation and evaluation for Learn mode."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from src.config import CHAT_MODEL
from src.curriculum import CurriculumModule
from src.openai_client import client
from src.retrieve import RetrievedChunk


logger = logging.getLogger(__name__)

QUIZ_PASS_THRESHOLD = 2 / 3
QUIZ_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "quiz.txt"
QUIZ_EVALUATION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "quiz_evaluation.txt"
QUIZ_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "number_correct": {"type": "integer"},
        "total_questions": {"type": "integer"},
        "percentage": {"type": "number"},
        "passed": {"type": "boolean"},
        "question_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "correct": {"type": "boolean"},
                    "explanation": {"type": "string"},
                },
                "required": ["id", "correct", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["number_correct", "total_questions", "percentage", "passed", "question_feedback"],
    "additionalProperties": False,
}


class QuizGenerationError(RuntimeError):
    """Raised when quiz generation fails."""


class QuizEvaluationError(RuntimeError):
    """Raised when quiz evaluation fails."""


@dataclass(slots=True)
class QuizQuestion:
    """A single curriculum quiz question."""

    id: str
    question: str
    objective: str
    reference_answer: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QuizQuestionFeedback:
    """Evaluation feedback for a single quiz question."""

    id: str
    correct: bool
    explanation: str


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
    number_correct: int
    total_questions: int
    percentage: float
    passed: bool
    question_feedback: list[QuizQuestionFeedback]

    @property
    def score(self) -> float:
        """Backward-compatible normalized score."""

        if self.total_questions <= 0:
            return 0.0
        return self.number_correct / self.total_questions

    @property
    def feedback(self) -> str:
        """Backward-compatible overall feedback summary."""

        if not self.question_feedback:
            return ""
        incorrect_ids = [item.id.upper() for item in self.question_feedback if not item.correct]
        if not incorrect_ids:
            return "All answers were correct."
        return f"Review questions: {', '.join(incorrect_ids)}"


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


def _build_lesson_context(
    lesson_title: str | None,
    lesson_content: str | None,
    lesson_takeaways: Sequence[str] | None,
) -> str:
    """Format the generated lesson content for quiz prompting."""

    sections: list[str] = []
    if lesson_title:
        sections.append(f"Lesson title: {lesson_title}")
    if lesson_content:
        sections.append("Lesson content:")
        sections.append(lesson_content.strip())
    if lesson_takeaways:
        sections.append("Key takeaways:")
        sections.extend(f"- {item}" for item in lesson_takeaways if str(item).strip())
    return "\n".join(sections).strip()


def _load_prompt_template() -> str:
    """Load the quiz-generation prompt from disk."""

    try:
        return QUIZ_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise QuizGenerationError(f"Prompt file not found: {QUIZ_PROMPT_PATH}") from exc


@lru_cache(maxsize=1)
def _get_prompt_template() -> str:
    return _load_prompt_template()


def _load_quiz_evaluation_prompt_template() -> str:
    """Load the quiz-evaluation prompt from disk."""

    try:
        return QUIZ_EVALUATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise QuizEvaluationError(f"Prompt file not found: {QUIZ_EVALUATION_PROMPT_PATH}") from exc


@lru_cache(maxsize=1)
def _get_quiz_evaluation_prompt_template() -> str:
    return _load_quiz_evaluation_prompt_template()


def _quiz_evaluation_response_format() -> dict[str, Any]:
    """Build the structured-output request for quiz evaluation."""

    return {
        "type": "json_schema",
        "name": "quiz_evaluation",
        "schema": QUIZ_EVALUATION_SCHEMA,
        "strict": True,
    }


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


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuizGenerationError(f"Quiz question {field_name} must be a non-empty string")
    return value.strip()


def _question_diagnostics(item: Any, fallback_id: str) -> tuple[str, str, list[Any]]:
    """Extract question diagnostics for logging without changing validation."""

    if isinstance(item, dict):
        raw_id = item.get("id")
        question_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else fallback_id
        objective = item.get("objective", "")
        source_chunk_ids = item.get("source_chunk_ids", [])
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []
        return question_id, str(objective), list(source_chunk_ids)

    return fallback_id, "", []


def _log_question_rejection(
    *,
    question_id: str,
    reason: str,
    objective: str,
    source_chunk_ids: list[Any],
) -> None:
    """Log a rejected quiz question for temporary diagnostics."""

    logger.warning(
        "Rejected quiz question id=%s reason=%s objective=%r source_chunk_ids=%r",
        question_id,
        reason,
        objective,
        source_chunk_ids,
    )


def _quiz_source_chunk_ids(retrieved_chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Build deterministic source chunk IDs from the retrieved quiz context."""

    source_chunk_ids: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        chunk_id = chunk.id.strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        source_chunk_ids.append(chunk_id)
    return source_chunk_ids


def _as_question(item: Any, fallback_id: str, *, allowed_objectives: set[str]) -> QuizQuestion:
    if isinstance(item, dict):
        question_id = _require_non_empty_string(item.get("id"), "id")
        question = _require_non_empty_string(item.get("question"), "question")
        objective = _require_non_empty_string(item.get("objective"), "objective")
        reference_answer = _require_non_empty_string(item.get("reference_answer"), "reference_answer")
        if objective not in allowed_objectives:
            raise QuizGenerationError("Quiz question objective must match a curriculum objective")
        return QuizQuestion(
            id=question_id.strip(),
            question=question,
            objective=objective,
            reference_answer=reference_answer,
            source_chunk_ids=[],
        )
    raise QuizGenerationError("Quiz questions must be returned as objects")


def generate_quiz(
    module: CurriculumModule,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    question_count: int = 5,
    lesson_title: str | None = None,
    lesson_content: str | None = None,
    lesson_takeaways: Sequence[str] | None = None,
) -> GeneratedQuiz:
    """Generate a short quiz from retrieved curriculum context."""

    if not retrieved_chunks:
        raise QuizGenerationError("Cannot generate a quiz without retrieved curriculum context")
    if question_count <= 0:
        raise QuizGenerationError("Question count must be a positive integer")

    context = _build_context(retrieved_chunks)
    lesson_context = _build_lesson_context(lesson_title, lesson_content, lesson_takeaways)
    prompt_template = _get_prompt_template()
    module_objectives = "\n".join(f"- {objective}" for objective in module.objectives)
    prompt = prompt_template.format(
        module_title=module.title,
        module_objectives=module_objectives,
        question_count=question_count,
        lesson_context=lesson_context,
        context=context,
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
    logger.warning("Quiz generation returned %d raw questions for module %s", len(raw_questions), module.id)

    allowed_objectives = {objective.strip() for objective in module.objectives if objective.strip()}
    source_chunk_ids = _quiz_source_chunk_ids(retrieved_chunks)
    questions: list[QuizQuestion] = []
    seen_question_ids: set[str] = set()
    for index, item in enumerate(raw_questions, start=1):
        if len(questions) >= question_count:
            break
        fallback_id = f"q{index}"
        question_id, objective, raw_source_chunk_ids = _question_diagnostics(item, fallback_id)
        try:
            question = _as_question(
                item,
                fallback_id,
                allowed_objectives=allowed_objectives,
            )
        except QuizGenerationError as exc:
            _log_question_rejection(
                question_id=question_id,
                reason=str(exc),
                objective=objective,
                source_chunk_ids=raw_source_chunk_ids,
            )
            continue

        if question.id in seen_question_ids:
            _log_question_rejection(
                question_id=question.id,
                reason="duplicate question id",
                objective=question.objective,
                source_chunk_ids=question.source_chunk_ids,
            )
            continue

        seen_question_ids.add(question.id)
        question.source_chunk_ids = list(source_chunk_ids)
        questions.append(question)

    if len(questions) < 2:
        raise QuizGenerationError("Quiz generation must return at least two valid questions")
    logger.warning("Quiz generation retained %d valid questions for module %s", len(questions), module.id)
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
    lesson_content: str,
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
    prompt_template = _get_quiz_evaluation_prompt_template()
    prompt = prompt_template.format(
        module_title=quiz.module_title,
        lesson_context=lesson_content.strip(),
        answers_payload=json.dumps(answers_payload, ensure_ascii=False),
        context=context,
    )

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
            text={"format": _quiz_evaluation_response_format()},
        )
    except Exception as exc:  # pragma: no cover - exercised via failure tests
        logger.exception("Quiz evaluation failed")
        raise QuizEvaluationError("Failed to evaluate the quiz") from exc

    output_text = getattr(response, "output_text", "")
    try:
        payload = _extract_json_object(output_text)
    except Exception as exc:  # pragma: no cover - guarded by tests
        raise QuizEvaluationError("Quiz evaluation must return JSON") from exc

    raw_question_feedback = payload.get("question_feedback", [])
    if not isinstance(raw_question_feedback, list):
        raise QuizEvaluationError("Quiz evaluation question feedback must be a list")

    question_feedback: list[QuizQuestionFeedback] = []
    for item in raw_question_feedback:
        if not isinstance(item, dict):
            raise QuizEvaluationError("Quiz evaluation question feedback must contain objects")

        feedback_id = str(item.get("id", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        correct = item.get("correct")
        if not feedback_id or not explanation or not isinstance(correct, bool):
            raise QuizEvaluationError("Quiz evaluation question feedback is incomplete")
        question_feedback.append(
            QuizQuestionFeedback(
                id=feedback_id,
                correct=correct,
                explanation=explanation,
            )
        )

    if len(question_feedback) != len(quiz.questions):
        raise QuizEvaluationError("Quiz evaluation must return feedback for each question")

    expected_question_ids = [question.id for question in quiz.questions]
    returned_question_ids = [item.id for item in question_feedback]
    if returned_question_ids != expected_question_ids:
        raise QuizEvaluationError("Quiz evaluation feedback ids must match the quiz questions")

    total_questions = len(question_feedback)
    if total_questions <= 0:
        raise QuizEvaluationError("Quiz evaluation question feedback must not be empty")

    number_correct = sum(1 for item in question_feedback if item.correct)
    percentage = round((number_correct / total_questions) * 100, 2)

    score_ratio = number_correct / total_questions
    passed = score_ratio >= QUIZ_PASS_THRESHOLD
    logger.info(
        "Evaluated quiz for module %s with %d/%d correct (%.2f%%) and passed=%s",
        quiz.module_id,
        number_correct,
        total_questions,
        percentage,
        passed,
    )
    return QuizEvaluation(
        module_id=quiz.module_id,
        number_correct=number_correct,
        total_questions=total_questions,
        percentage=percentage,
        passed=passed,
        question_feedback=question_feedback,
    )

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import src.quiz as quiz_module
from src.curriculum import get_module
from src.quiz import GeneratedQuiz, QuizQuestion, QuizQuestionFeedback
from src.retrieve import RetrievedChunk


def _make_chunk(
    *,
    title: str = "EMA Human Regulatory Overview",
    relative_file_path: str = "ema/ema_human_regulatory_overview.pdf",
    text: str = "Regulatory agencies oversee drug development and approval.",
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{relative_file_path}::chunk-0",
        score=0.91,
        text=text,
        metadata={
            "filename": relative_file_path.rsplit("/", 1)[-1],
            "relative_file_path": relative_file_path,
            "source_organization": relative_file_path.split("/", 1)[0],
            "document_title": title,
            "chunk_id": f"{relative_file_path}::chunk-0",
            "text": text,
        },
    )


def test_generate_quiz_uses_retrieved_curriculum_context() -> None:
    module = get_module("foundations")
    assert module is not None
    retrieved_chunks = [
        _make_chunk(),
        _make_chunk(
            title="FDA Drug Development and Approval Process",
            relative_file_path="fda/fda_drug_development_process.pdf",
            text="FDA guidance describes the drug development process.",
        ),
    ]
    payload = {
        "questions": [
            {
                "id": "q1",
                "question": "What is the purpose of Good Clinical Practice?",
                "objective": "Understand the drug development lifecycle",
                "reference_answer": "To provide standards for clinical trials.",
                "source_chunk_ids": [retrieved_chunks[0].id],
            },
            {
                "id": "q2",
                "question": "Name one regulatory agency involved in drug development.",
                "objective": "Recognize major stakeholders",
                "reference_answer": "FDA or EMA.",
                "source_chunk_ids": [retrieved_chunks[1].id],
            },
            {
                "id": "q3",
                "question": "What is the drug development lifecycle?",
                "objective": "Understand the drug development lifecycle",
                "reference_answer": "The stages from discovery to approval.",
                "source_chunk_ids": [retrieved_chunks[0].id, retrieved_chunks[1].id],
            },
        ]
    }
    captured: dict[str, str] = {}

    def fake_create(*, model: str, input: str):
        captured["input"] = input
        return SimpleNamespace(output_text=json.dumps(payload))

    with patch.object(quiz_module.client.responses, "create", side_effect=fake_create):
        quiz = quiz_module.generate_quiz(
            module,
            retrieved_chunks,
            lesson_title="Introduction to Drug Development",
            lesson_content="The lesson explains the lifecycle, stakeholders, and terminology.",
            lesson_takeaways=[
                "Understand the drug development lifecycle.",
                "Recognize major stakeholders.",
            ],
        )

    assert quiz.module_id == module.id
    assert quiz.module_title == module.title
    assert len(quiz.questions) == 3
    assert [question.objective for question in quiz.questions] == [
        "Understand the drug development lifecycle",
        "Recognize major stakeholders",
        "Understand the drug development lifecycle",
    ]
    assert [question.reference_answer for question in quiz.questions] == [
        "To provide standards for clinical trials.",
        "FDA or EMA.",
        "The stages from discovery to approval.",
    ]
    assert quiz.questions[0].source_chunk_ids == [retrieved_chunks[0].id]
    assert quiz.questions[1].source_chunk_ids == [retrieved_chunks[1].id]
    assert quiz.questions[2].source_chunk_ids == [retrieved_chunks[0].id, retrieved_chunks[1].id]
    assert quiz.source_document_titles == [
        "EMA Human Regulatory Overview",
        "FDA Drug Development and Approval Process",
    ]
    assert "Introduction to Drug Development" in captured["input"]
    assert "Understand the drug development lifecycle" in captured["input"]
    assert "The lesson explains the lifecycle, stakeholders, and terminology." in captured["input"]
    assert "Recognize major stakeholders." in captured["input"]
    assert "Lesson content:" in captured["input"]
    assert "Create exactly 5 questions" in captured["input"]
    assert retrieved_chunks[0].text in captured["input"]
    assert quiz_module._get_prompt_template().startswith("You are creating a knowledge check")


def test_generate_quiz_raises_when_prompt_file_is_missing(tmp_path: Path) -> None:
    module = get_module("foundations")
    assert module is not None
    retrieved_chunks = [_make_chunk()]
    missing_prompt = tmp_path / "missing_quiz_prompt.txt"

    quiz_module._get_prompt_template.cache_clear()
    with (
        patch.object(quiz_module, "QUIZ_PROMPT_PATH", missing_prompt),
        patch.object(quiz_module.client.responses, "create") as create,
    ):
        try:
            quiz_module.generate_quiz(module, retrieved_chunks)
        except quiz_module.QuizGenerationError as exc:
            assert "Prompt file not found" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("Expected QuizGenerationError")

    assert create.call_count == 0
    quiz_module._get_prompt_template.cache_clear()


def test_evaluate_quiz_is_grounded_in_context_and_returns_structured_result() -> None:
    quiz = GeneratedQuiz(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        questions=[
            QuizQuestion(
                id="q1",
                question="What is GCP?",
                objective="Understand the drug development lifecycle",
                reference_answer="Good Clinical Practice.",
            ),
            QuizQuestion(
                id="q2",
                question="Name one stakeholder.",
                objective="Recognize major stakeholders",
                reference_answer="Sponsor.",
            ),
            QuizQuestion(
                id="q3",
                question="What is drug development?",
                objective="Understand the drug development lifecycle",
                reference_answer="The process from discovery to approval.",
            ),
        ],
        source_document_titles=["EMA Human Regulatory Overview"],
        context_summary="context",
    )
    retrieved_chunks = [_make_chunk()]
    payload = {
        "number_correct": 2,
        "total_questions": 3,
        "percentage": 66.67,
        "passed": True,
        "question_feedback": [
            {"id": "q1", "correct": True, "explanation": "Good Clinical Practice is a standard for trials."},
            {"id": "q2", "correct": True, "explanation": "The learner identified a stakeholder."},
            {"id": "q3", "correct": False, "explanation": "The answer was too vague."},
        ],
    }
    captured: dict[str, object] = {}

    def fake_create(*, model: str, input: str, text=None):
        captured["input"] = input
        captured["text"] = text
        return SimpleNamespace(output_text=json.dumps(payload))

    with patch.object(quiz_module.client.responses, "create", side_effect=fake_create):
        result = quiz_module.evaluate_quiz(
            quiz,
            ["Good Clinical Practice", "Sponsor", "Drug lifecycle"],
            retrieved_chunks,
        )

    assert result.module_id == "foundations"
    assert result.number_correct == 2
    assert result.total_questions == 3
    assert result.percentage == 66.67
    assert result.passed is True
    assert quiz_module.QUIZ_PASS_THRESHOLD == 2 / 3
    assert result.question_feedback == [
        QuizQuestionFeedback(
            id="q1",
            correct=True,
            explanation="Good Clinical Practice is a standard for trials.",
        ),
        QuizQuestionFeedback(
            id="q2",
            correct=True,
            explanation="The learner identified a stakeholder.",
        ),
        QuizQuestionFeedback(
            id="q3",
            correct=False,
            explanation="The answer was too vague.",
        ),
    ]
    assert "Good Clinical Practice" in captured["input"]
    assert retrieved_chunks[0].text in captured["input"]
    assert "Return strict JSON with fields: number_correct, total_questions, percentage, passed, question_feedback." in captured["input"]
    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "quiz_evaluation",
            "schema": quiz_module.QUIZ_EVALUATION_SCHEMA,
            "strict": True,
        }
    }

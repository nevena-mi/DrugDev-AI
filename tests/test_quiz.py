from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import src.quiz as quiz_module
from src.curriculum import get_module
from src.quiz import GeneratedQuiz, QuizQuestion
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
                "reference_answer": "To provide standards for clinical trials.",
                "source_chunk_ids": [retrieved_chunks[0].id],
            },
            {
                "id": "q2",
                "question": "Name one regulatory agency involved in drug development.",
                "reference_answer": "FDA or EMA.",
                "source_chunk_ids": [retrieved_chunks[1].id],
            },
            {
                "id": "q3",
                "question": "What is the drug development lifecycle?",
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
        quiz = quiz_module.generate_quiz(module, retrieved_chunks)

    assert quiz.module_id == module.id
    assert quiz.module_title == module.title
    assert len(quiz.questions) == 3
    assert quiz.source_document_titles == [
        "EMA Human Regulatory Overview",
        "FDA Drug Development and Approval Process",
    ]
    assert "Introduction to Drug Development" in captured["input"]
    assert retrieved_chunks[0].text in captured["input"]


def test_evaluate_quiz_is_grounded_in_context_and_returns_structured_result() -> None:
    quiz = GeneratedQuiz(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        questions=[
            QuizQuestion(
                id="q1",
                question="What is GCP?",
                reference_answer="Good Clinical Practice.",
            ),
            QuizQuestion(
                id="q2",
                question="Name one stakeholder.",
                reference_answer="Sponsor.",
            ),
            QuizQuestion(
                id="q3",
                question="What is drug development?",
                reference_answer="The process from discovery to approval.",
            ),
        ],
        source_document_titles=["EMA Human Regulatory Overview"],
        context_summary="context",
    )
    retrieved_chunks = [_make_chunk()]
    payload = {
        "score": 0.67,
        "passed": True,
        "feedback": "Good work.",
        "question_feedback": ["q1 ok", "q2 ok", "q3 ok"],
    }
    captured: dict[str, str] = {}

    def fake_create(*, model: str, input: str):
        captured["input"] = input
        return SimpleNamespace(output_text=json.dumps(payload))

    with patch.object(quiz_module.client.responses, "create", side_effect=fake_create):
        result = quiz_module.evaluate_quiz(
            quiz,
            ["Good Clinical Practice", "Sponsor", "Drug lifecycle"],
            retrieved_chunks,
        )

    assert result.module_id == "foundations"
    assert result.score == 0.67
    assert result.passed is True
    assert result.feedback == "Good work."
    assert result.question_feedback == ["q1 ok", "q2 ok", "q3 ok"]
    assert "Good Clinical Practice" in captured["input"]
    assert retrieved_chunks[0].text in captured["input"]


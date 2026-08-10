from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import src.graph as graph_module
from src.curriculum import get_module
from src.retrieve import RetrievedChunk


def _make_chunk(
    *,
    title: str = "ICH E6(R3) Guideline for Good Clinical Practice",
    relative_file_path: str = "ich/ich_e6_r3.pdf",
    text: str = "Good Clinical Practice sets standards for clinical trials.",
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


def test_onboarding_returns_a_valid_predefined_module() -> None:
    profile = graph_module.LearnerProfile(
        academic_or_professional_background="Scientist",
        drug_development_familiarity="Beginner",
        learning_goal="Learn the basics",
        prior_regulatory_pharma_experience="None",
        available_study_time="3 hours/week",
    )

    with patch.object(
        graph_module.client.responses,
        "create",
        return_value=SimpleNamespace(output_text="not-a-module"),
    ):
        session = graph_module.start_learning_session(profile)

    assert session.recommended_start_module_id == "foundations"
    assert session.current_module_id == "foundations"


def test_lesson_generation_uses_active_module_documents_and_context() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [
        _make_chunk(
            title="ICH E6(R3) Guideline for Good Clinical Practice",
            relative_file_path="ich/ich_e6_r3.pdf",
            text="Good Clinical Practice sets standards for clinical trials.",
        ),
        _make_chunk(
            title="Declaration of Helsinki",
            relative_file_path="wma/declaration_of_helsinki.pdf",
            text="Ethical principles apply to human subjects research.",
        ),
    ]
    payload = {
        "lesson_title": "Clinical Trials and GCP",
        "learning_content": "Clinical trials rely on ethical conduct, protocol adherence, and GCP.",
        "key_takeaways": [
            "Clinical trials must follow Good Clinical Practice.",
            "Ethical principles protect participants.",
            "Module documents should guide the study plan.",
        ],
    }
    captured: dict[str, str] = {}
    document_paths_seen: list[list[str] | None] = []

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = 5,
        namespace=None,
        document_paths=None,
    ):
        captured["query"] = query
        document_paths_seen.append(None if document_paths is None else list(document_paths))
        assert list(document_paths or []) == module.documents
        return retrieved_chunks

    def fake_create(**kwargs):
        captured["prompt"] = kwargs["input"]
        captured["text"] = kwargs["text"]
        return SimpleNamespace(output_text=json.dumps(payload))

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert lesson.module_id == module.id
    assert lesson.lesson_title == payload["lesson_title"]
    assert lesson.learning_content == payload["learning_content"]
    assert lesson.key_takeaways == payload["key_takeaways"]
    assert [citation.document_title for citation in lesson.citations] == [
        "ICH E6(R3) Guideline for Good Clinical Practice",
        "Declaration of Helsinki",
    ]
    assert document_paths_seen == [module.documents]
    assert module.title in captured["prompt"]
    assert module.description in captured["prompt"]
    for objective in module.objectives:
        assert objective in captured["prompt"]
    assert retrieved_chunks[0].text in captured["prompt"]
    assert retrieved_chunks[1].text in captured["prompt"]
    assert "return strict json only" in captured["prompt"].lower()
    assert "lesson_title" in captured["prompt"]


def test_lesson_generation_requests_structured_output_schema() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [_make_chunk()]
    captured: dict[str, object] = {}

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = 5,
        namespace=None,
        document_paths=None,
    ):
        return retrieved_chunks

    def fake_create(*, model: str, input: str, text=None):
        captured["model"] = model
        captured["input"] = input
        captured["text"] = text
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "lesson_title": "Clinical Trials and GCP",
                    "learning_content": "Clinical trials rely on ethical conduct.",
                    "key_takeaways": ["One", "Two", "Three"],
                }
            )
        )

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        graph_module.generate_learning_lesson("clinical_trials")

    assert captured["model"] == graph_module.CHAT_MODEL
    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "lesson_generation",
            "schema": graph_module.LESSON_RESPONSE_SCHEMA,
            "strict": True,
        }
    }
    assert captured["input"] is not None


def test_lesson_generation_returns_safe_result_when_context_is_missing() -> None:
    module = get_module("clinical_trials")
    assert module is not None

    with (
        patch.object(graph_module, "retrieve_chunks", return_value=[]),
        patch.object(graph_module.client.responses, "create") as create,
    ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert create.call_count == 0
    assert lesson.learning_content == graph_module.LESSON_INSUFFICIENT_INFORMATION
    assert lesson.key_takeaways == []
    assert lesson.citations == []


def test_lesson_generation_rejects_malformed_json() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [_make_chunk()]

    with (
        patch.object(graph_module, "retrieve_chunks", return_value=retrieved_chunks),
        patch.object(
            graph_module.client.responses,
            "create",
            return_value=SimpleNamespace(output_text="not-json"),
        ),
    ):
        try:
            graph_module.generate_learning_lesson("clinical_trials")
        except graph_module.LessonGenerationError:
            pass
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("Expected LessonGenerationError")


def test_lesson_session_cache_reuses_existing_lesson_until_module_changes() -> None:
    profile = graph_module.LearnerProfile(
        academic_or_professional_background="Scientist",
        drug_development_familiarity="Beginner",
        learning_goal="Learn the basics",
        prior_regulatory_pharma_experience="None",
        available_study_time="3 hours/week",
    )
    session = graph_module.LearningSession(
        profile=profile,
        recommended_start_module_id="foundations",
        current_module_id="foundations",
    )
    first_lesson = graph_module.LearningLesson(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        lesson_title="Lesson 1",
        learning_content="Content 1",
        key_takeaways=["A"],
        citations=[],
        retrieved_chunks=[],
        retrieval_scope="module",
    )
    second_lesson = graph_module.LearningLesson(
        module_id="regulatory",
        module_title="Regulatory Landscape",
        lesson_title="Lesson 2",
        learning_content="Content 2",
        key_takeaways=["B"],
        citations=[],
        retrieved_chunks=[],
        retrieval_scope="module",
    )

    with patch.object(
        graph_module,
        "generate_learning_lesson",
        side_effect=[first_lesson, second_lesson],
    ) as generate:
        lesson_one = graph_module.ensure_learning_lesson(session)
        lesson_two = graph_module.ensure_learning_lesson(session)
        session.current_module_id = "regulatory"
        lesson_three = graph_module.ensure_learning_lesson(session)

    assert lesson_one is first_lesson
    assert lesson_two is first_lesson
    assert lesson_three is second_lesson
    assert session.current_lesson is second_lesson
    assert generate.call_count == 2


def test_module_scoped_learning_question_uses_strict_documents_then_fallback() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    fallback_chunk = _make_chunk()
    call_document_paths: list[list[str] | None] = []

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = 5,
        namespace=None,
        document_paths=None,
    ):
        call_document_paths.append(None if document_paths is None else list(document_paths))
        if document_paths is not None:
            assert list(document_paths) == module.documents
            return []
        return [fallback_chunk]

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(
            graph_module.client.responses,
            "create",
            return_value=SimpleNamespace(output_text="Grounded answer"),
        ) as create,
    ):
        answer = graph_module.answer_learning_question("clinical_trials", "What is GCP?")

    assert call_document_paths == [module.documents, None]
    assert answer.retrieval_scope == "fallback"
    assert answer.answer == "Grounded answer"
    assert answer.citations[0].document_title == fallback_chunk.metadata["document_title"]
    prompt = create.call_args.kwargs["input"]
    assert "What is GCP?" in prompt
    assert fallback_chunk.text in prompt


def test_learning_progression_respects_prerequisites() -> None:
    profile = graph_module.LearnerProfile(
        academic_or_professional_background="Scientist",
        drug_development_familiarity="Beginner",
        learning_goal="Learn the basics",
        prior_regulatory_pharma_experience="None",
        available_study_time="3 hours/week",
    )
    session = graph_module.LearningSession(
        profile=profile,
        recommended_start_module_id="foundations",
        current_module_id="foundations",
    )

    assert graph_module.preview_next_module(session).id == "regulatory"
    assert graph_module.recommend_next_module(["foundations"]).id == "regulatory"

    next_module = graph_module.complete_current_module(session)

    assert next_module is not None
    assert next_module.id == "regulatory"
    assert session.completed_module_ids == ["foundations"]
    assert session.current_module_id == "regulatory"

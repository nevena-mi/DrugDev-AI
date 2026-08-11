from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import src.graph as graph_module
from src.curriculum import get_module
from src.quiz import GeneratedQuiz
from src.rerank import RerankedChunk
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


def _make_reranked_chunk(
    chunk: RetrievedChunk,
    *,
    cohere_score: float,
    original_index: int,
    reranked_rank: int,
) -> RerankedChunk:
    return RerankedChunk(
        id=chunk.id,
        text=chunk.text,
        metadata=dict(chunk.metadata),
        pinecone_score=chunk.score,
        cohere_score=cohere_score,
        original_index=original_index,
        reranked_rank=reranked_rank,
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


def test_lesson_generation_uses_active_module_documents_and_reranked_context() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [
        _make_chunk(
            title="ICH E6(R3) Guideline for Good Clinical Practice",
            relative_file_path="ich/ich_e6_r3.pdf",
            text="Chunk one about GCP and clinical trial conduct.",
        ),
        _make_chunk(
            title="Declaration of Helsinki",
            relative_file_path="wma/declaration_of_helsinki.pdf",
            text="Chunk two about human subject protections.",
        ),
        _make_chunk(
            title="ICH M3(R2) Nonclinical Safety Studies",
            relative_file_path="ich/ich_m3_r2.pdf",
            text="Chunk three about nonclinical safety.",
        ),
        _make_chunk(
            title="FDA Drug Development and Approval Process",
            relative_file_path="fda/fda_drug_development_process.pdf",
            text="Chunk four about the development pipeline.",
        ),
        _make_chunk(
            title="WHO Good Manufacturing Practices for Pharmaceutical Products",
            relative_file_path="who/who_gmp.pdf",
            text="Chunk five about manufacturing quality.",
        ),
        _make_chunk(
            title="ICH Q9 Quality Risk Management",
            relative_file_path="ich/ich_q9.pdf",
            text="Chunk six about risk management.",
        ),
    ]
    reranked_chunks = [
        _make_reranked_chunk(retrieved_chunks[5], cohere_score=0.99, original_index=5, reranked_rank=1),
        _make_reranked_chunk(retrieved_chunks[3], cohere_score=0.98, original_index=3, reranked_rank=2),
        _make_reranked_chunk(retrieved_chunks[1], cohere_score=0.97, original_index=1, reranked_rank=3),
        _make_reranked_chunk(retrieved_chunks[0], cohere_score=0.96, original_index=0, reranked_rank=4),
        _make_reranked_chunk(retrieved_chunks[2], cohere_score=0.95, original_index=2, reranked_rank=5),
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
    captured: dict[str, object] = {}
    document_paths_seen: list[list[str] | None] = []

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        captured.setdefault("queries", []).append(query)
        captured.setdefault("top_ks", []).append(top_k)
        document_paths_seen.append(None if document_paths is None else list(document_paths))
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        assert list(document_paths or []) == module.documents
        return retrieved_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        captured["rerank_question"] = question
        captured["rerank_top_n"] = top_n
        captured["rerank_texts"] = [chunk.text for chunk in chunks]
        return reranked_chunks

    def fake_create(**kwargs):
        captured["prompt"] = kwargs["input"]
        captured["text"] = kwargs["text"]
        return SimpleNamespace(output_text=json.dumps(payload))

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert lesson.module_id == module.id
    assert lesson.lesson_title == payload["lesson_title"]
    assert lesson.learning_content == payload["learning_content"]
    assert lesson.key_takeaways == payload["key_takeaways"]
    assert [chunk.id for chunk in lesson.retrieved_chunks] == [chunk.id for chunk in reranked_chunks]
    assert [citation.document_title for citation in lesson.citations] == [
        "ICH Q9 Quality Risk Management",
        "FDA Drug Development and Approval Process",
        "Declaration of Helsinki",
        "ICH E6(R3) Guideline for Good Clinical Practice",
        "ICH M3(R2) Nonclinical Safety Studies",
    ]
    assert document_paths_seen == [module.documents]
    assert captured["rerank_top_n"] == 5
    assert captured["rerank_question"] == " ".join(
        [
            module.title,
            module.description,
            "; ".join(module.objectives),
        ]
    )
    assert captured["rerank_texts"] == [chunk.text for chunk in retrieved_chunks]
    assert module.title in captured["prompt"]
    assert module.description in captured["prompt"]
    for objective in module.objectives:
        assert objective in captured["prompt"]
    assert reranked_chunks[0].text in captured["prompt"]
    assert reranked_chunks[4].text in captured["prompt"]
    assert retrieved_chunks[4].text not in captured["prompt"]
    assert "return strict json only" in str(captured["prompt"]).lower()
    assert "lesson_title" in str(captured["prompt"])
    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "lesson_generation",
            "schema": graph_module.LESSON_RESPONSE_SCHEMA,
            "strict": True,
        }
    }


def test_lesson_generation_requests_structured_output_schema() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [_make_chunk()]

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        return retrieved_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        return [_make_reranked_chunk(chunk, cohere_score=0.9, original_index=index, reranked_rank=index + 1) for index, chunk in enumerate(chunks[:top_n])]

    captured: dict[str, object] = {}

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
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
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
        patch.object(graph_module, "rerank_chunks") as rerank,
        patch.object(graph_module.client.responses, "create") as create,
    ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert rerank.call_count == 0
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
            graph_module,
            "rerank_chunks",
            return_value=[
                _make_reranked_chunk(retrieved_chunks[0], cohere_score=0.9, original_index=0, reranked_rank=1)
            ],
        ),
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


def test_lesson_generation_reranking_failure_falls_back_to_pinecone_top_five() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    retrieved_chunks = [
        _make_chunk(text=f"Chunk {index}", relative_file_path=f"ich/doc-{index}.pdf", title=f"Document {index}")
        for index in range(1, 7)
    ]
    captured: dict[str, object] = {}

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        assert list(document_paths or []) == module.documents
        return retrieved_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        raise graph_module.RerankingError("Cohere unavailable")

    def fake_create(**kwargs):
        captured["prompt"] = kwargs["input"]
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
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(graph_module.client.responses, "create", side_effect=fake_create),
    ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert [chunk.id for chunk in lesson.retrieved_chunks] == [chunk.id for chunk in retrieved_chunks[:5]]
    assert lesson.retrieval_scope == "module"
    assert all(chunk.text in str(captured["prompt"]) for chunk in retrieved_chunks[:5])
    assert retrieved_chunks[5].text not in str(captured["prompt"])


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
    fallback_chunks = [
        _make_chunk(
            title="ICH E6(R3) Guideline for Good Clinical Practice",
            relative_file_path="ich/ich_e6_r3.pdf",
            text="Fallback chunk one about GCP.",
        ),
        _make_chunk(
            title="Declaration of Helsinki",
            relative_file_path="wma/declaration_of_helsinki.pdf",
            text="Fallback chunk two about ethics.",
        ),
        _make_chunk(
            title="ICH M3(R2) Nonclinical Safety Studies",
            relative_file_path="ich/ich_m3_r2.pdf",
            text="Fallback chunk three about nonclinical safety.",
        ),
        _make_chunk(
            title="FDA Drug Development and Approval Process",
            relative_file_path="fda/fda_drug_development_process.pdf",
            text="Fallback chunk four about the development pipeline.",
        ),
        _make_chunk(
            title="WHO Good Manufacturing Practices for Pharmaceutical Products",
            relative_file_path="who/who_gmp.pdf",
            text="Fallback chunk five about manufacturing quality.",
        ),
        _make_chunk(
            title="ICH Q9 Quality Risk Management",
            relative_file_path="ich/ich_q9.pdf",
            text="Fallback chunk six about risk management.",
        ),
    ]
    reranked_chunks = [
        _make_reranked_chunk(fallback_chunks[4], cohere_score=0.99, original_index=4, reranked_rank=1),
        _make_reranked_chunk(fallback_chunks[2], cohere_score=0.98, original_index=2, reranked_rank=2),
        _make_reranked_chunk(fallback_chunks[0], cohere_score=0.97, original_index=0, reranked_rank=3),
        _make_reranked_chunk(fallback_chunks[1], cohere_score=0.96, original_index=1, reranked_rank=4),
        _make_reranked_chunk(fallback_chunks[3], cohere_score=0.95, original_index=3, reranked_rank=5),
    ]
    call_document_paths: list[list[str] | None] = []
    captured: dict[str, object] = {}

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        call_document_paths.append(None if document_paths is None else list(document_paths))
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        if document_paths is not None:
            assert list(document_paths) == module.documents
            return []
        return fallback_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        captured["rerank_question"] = question
        captured["rerank_texts"] = [chunk.text for chunk in chunks]
        captured["rerank_top_n"] = top_n
        return reranked_chunks

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(
            graph_module.client.responses,
            "create",
            return_value=SimpleNamespace(output_text="Grounded answer"),
        ) as create,
    ):
        answer = graph_module.answer_learning_question("clinical_trials", "What is GCP?")

    assert call_document_paths == [module.documents, None]
    assert captured["rerank_top_n"] == 5
    assert answer.retrieval_scope == "fallback"
    assert answer.answer == "Grounded answer"
    assert [chunk.id for chunk in answer.retrieved_chunks] == [chunk.id for chunk in reranked_chunks]
    assert [citation.document_title for citation in answer.citations] == [
        fallback_chunks[4].metadata["document_title"],
        fallback_chunks[2].metadata["document_title"],
        fallback_chunks[0].metadata["document_title"],
        fallback_chunks[1].metadata["document_title"],
        fallback_chunks[3].metadata["document_title"],
    ]
    prompt = create.call_args.kwargs["input"]
    assert "What is GCP?" in prompt
    assert fallback_chunks[4].text in prompt
    assert fallback_chunks[3].text in prompt
    assert fallback_chunks[5].text not in prompt


def test_module_scoped_learning_question_reranking_failure_falls_back_to_pinecone_top_five() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    strict_chunks = [
        _make_chunk(text=f"Strict chunk {index}", relative_file_path=f"ich/strict-{index}.pdf", title=f"Strict {index}")
        for index in range(1, 7)
    ]
    call_document_paths: list[list[str] | None] = []

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        call_document_paths.append(None if document_paths is None else list(document_paths))
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        assert list(document_paths or []) == module.documents
        return strict_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        raise graph_module.RerankingError("Cohere unavailable")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(graph_module, "_get_prompt_template", return_value="{question}\n{context}"),
        patch.object(
            graph_module.client.responses,
            "create",
            return_value=SimpleNamespace(output_text="Grounded answer"),
        ),
    ):
        answer = graph_module.answer_learning_question("clinical_trials", "What is GCP?")

    assert call_document_paths == [module.documents]
    assert answer.retrieval_scope == "module"
    assert [chunk.id for chunk in answer.retrieved_chunks] == [chunk.id for chunk in strict_chunks[:5]]


def test_learning_quiz_uses_strict_documents_and_reranked_context() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    strict_chunks = [
        _make_chunk(
            title=f"Module Document {index}",
            relative_file_path=f"ich/module-{index}.pdf",
            text=f"Strict quiz chunk {index}",
        )
        for index in range(1, 7)
    ]
    reranked_chunks = [
        _make_reranked_chunk(strict_chunks[4], cohere_score=0.99, original_index=4, reranked_rank=1),
        _make_reranked_chunk(strict_chunks[3], cohere_score=0.98, original_index=3, reranked_rank=2),
        _make_reranked_chunk(strict_chunks[2], cohere_score=0.97, original_index=2, reranked_rank=3),
        _make_reranked_chunk(strict_chunks[1], cohere_score=0.96, original_index=1, reranked_rank=4),
        _make_reranked_chunk(strict_chunks[0], cohere_score=0.95, original_index=0, reranked_rank=5),
    ]
    captured: dict[str, object] = {}

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        captured.setdefault("retrieval_queries", []).append(query)
        captured.setdefault("retrieval_top_ks", []).append(top_k)
        captured.setdefault("retrieval_paths", []).append(None if document_paths is None else list(document_paths))
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        assert list(document_paths or []) == module.documents
        return strict_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        captured["rerank_question"] = question
        captured["rerank_texts"] = [chunk.text for chunk in chunks]
        captured["rerank_top_n"] = top_n
        return reranked_chunks

    def fake_generate_quiz(module_obj, retrieved_chunks, *, question_count: int = 5, lesson_title=None, lesson_content=None, lesson_takeaways=None):
        captured["quiz_chunks"] = [chunk.id for chunk in retrieved_chunks]
        captured["quiz_question_count"] = question_count
        captured["lesson_title"] = lesson_title
        captured["lesson_content"] = lesson_content
        captured["lesson_takeaways"] = list(lesson_takeaways or [])
        return GeneratedQuiz(
            module_id=module_obj.id,
            module_title=module_obj.title,
            questions=[],
            source_document_titles=[],
            context_summary="",
        )

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(graph_module, "generate_quiz", side_effect=fake_generate_quiz),
    ):
        bundle = graph_module.generate_learning_quiz("clinical_trials")

    assert captured["retrieval_top_ks"] == [graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K]
    assert captured["retrieval_paths"] == [module.documents]
    assert captured["rerank_top_n"] == 5
    assert bundle.retrieval_scope == "module"
    assert [chunk.id for chunk in bundle.retrieved_chunks] == [chunk.id for chunk in reranked_chunks]
    assert captured["quiz_chunks"] == [chunk.id for chunk in reranked_chunks]


def test_lesson_generation_reranking_failure_falls_back_to_module_pinecone_top_five() -> None:
    module = get_module("clinical_trials")
    assert module is not None
    strict_chunks = [
        _make_chunk(text=f"Strict lesson chunk {index}", relative_file_path=f"ich/lesson-{index}.pdf", title=f"Lesson {index}")
        for index in range(1, 7)
    ]

    def fake_retrieve_chunks(
        query: str,
        *,
        top_k: int = graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K,
        namespace=None,
        document_paths=None,
    ):
        assert top_k == graph_module.LEARN_RETRIEVAL_CANDIDATE_TOP_K
        assert list(document_paths or []) == module.documents
        return strict_chunks

    def fake_rerank(question: str, chunks, *, top_n: int = 5, model: str | None = None):
        raise graph_module.RerankingError("Cohere unavailable")

    with (
        patch.object(graph_module, "retrieve_chunks", side_effect=fake_retrieve_chunks),
        patch.object(graph_module, "rerank_chunks", side_effect=fake_rerank),
        patch.object(
            graph_module.client.responses,
            "create",
            return_value=SimpleNamespace(
                output_text=json.dumps(
                    {
                        "lesson_title": "Clinical Trials and GCP",
                        "learning_content": "Clinical trials rely on ethical conduct.",
                        "key_takeaways": ["One", "Two", "Three"],
                    }
                )
            ),
        ) as create,
        ):
        lesson = graph_module.generate_learning_lesson("clinical_trials")

    assert create.call_count == 1
    assert [chunk.id for chunk in lesson.retrieved_chunks] == [chunk.id for chunk in strict_chunks[:5]]
    assert lesson.retrieval_scope == "module"


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

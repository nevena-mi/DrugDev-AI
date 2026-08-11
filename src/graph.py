"""Ask and Learn workflow orchestration for grounded RAG."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from src.config import CHAT_MODEL
from src.curriculum import (
    CurriculumModule,
    earliest_entry_module,
    get_module,
    next_modules as curriculum_next_modules,
)
from src.openai_client import client
from src.rerank import RerankedChunk, RerankingError, rerank_chunks
from src.quiz import GeneratedQuiz, QuizEvaluation, evaluate_quiz, generate_quiz
from src.retrieve import RetrievedChunk, retrieve_chunks


logger = logging.getLogger(__name__)

INSUFFICIENT_INFORMATION = "I cannot answer from the available documents."
ASK_RETRIEVAL_CANDIDATE_TOP_K = 15
ASK_FINAL_TOP_K = 5
LEARN_RETRIEVAL_CANDIDATE_TOP_K = 15
LEARN_FINAL_TOP_K = 5
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "answer.txt"
LESSON_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "lesson.txt"
LESSON_INSUFFICIENT_INFORMATION = "I cannot generate a lesson from the available documents."
LESSON_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson_title": {"type": "string"},
        "learning_content": {"type": "string"},
        "key_takeaways": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["lesson_title", "learning_content", "key_takeaways"],
    "additionalProperties": False,
}


class RAGGenerationError(RuntimeError):
    """Raised when the answer-generation step fails."""


class LessonGenerationError(RuntimeError):
    """Raised when lesson generation fails."""


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
    document_paths: list[str] | None = None
    allow_fallback: bool = False
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    context: str = ""
    llm_output: str = ""
    answer: str = ""
    citations: list[SourceCitation] = field(default_factory=list)
    retrieval_scope: str = "corpus"


@dataclass(slots=True)
class RAGResult:
    """Structured response returned by the Ask workflow."""

    question: str
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]


@dataclass(slots=True)
class LearnerProfile:
    """Session-level learner profile for Learn mode."""

    academic_or_professional_background: str
    drug_development_familiarity: str
    learning_goal: str
    prior_regulatory_pharma_experience: str
    available_study_time: str


@dataclass(slots=True)
class LearningAnswer:
    """Grounded learning response for a module question."""

    module_id: str
    module_title: str
    question: str
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_scope: str


@dataclass(slots=True)
class LearningQuizBundle:
    """Generated quiz package for a curriculum module."""

    module_id: str
    module_title: str
    quiz: GeneratedQuiz
    retrieved_chunks: list[RetrievedChunk]
    retrieval_scope: str


@dataclass(slots=True)
class LearningLesson:
    """Structured lesson content for a curriculum module."""

    module_id: str
    module_title: str
    lesson_title: str
    learning_content: str
    key_takeaways: list[str]
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_scope: str


@dataclass(slots=True)
class LearningSession:
    """Mutable learning session stored in Streamlit state."""

    profile: LearnerProfile
    recommended_start_module_id: str
    current_module_id: str
    completed_module_ids: list[str] = field(default_factory=list)
    quiz_result: QuizEvaluation | None = None
    current_quiz: LearningQuizBundle | None = None
    current_lesson: LearningLesson | None = None


def _load_prompt_template() -> str:
    """Load the answer-generation prompt from disk."""

    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RAGGenerationError(f"Prompt file not found: {PROMPT_PATH}") from exc


@lru_cache(maxsize=1)
def _get_prompt_template() -> str:
    return _load_prompt_template()


def _load_lesson_prompt_template() -> str:
    """Load the lesson-generation prompt from disk."""

    try:
        return LESSON_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise LessonGenerationError(f"Prompt file not found: {LESSON_PROMPT_PATH}") from exc


@lru_cache(maxsize=1)
def _get_lesson_prompt_template() -> str:
    return _load_lesson_prompt_template()


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


def _reranked_to_retrieved_chunk(chunk: RerankedChunk) -> RetrievedChunk:
    """Convert a reranked chunk back into the retrieval chunk shape."""

    return RetrievedChunk(
        id=chunk.id,
        score=chunk.pinecone_score,
        text=chunk.text,
        metadata=dict(chunk.metadata),
    )


def _select_ask_chunks(question: str, *, namespace: str | None = None) -> tuple[list[RetrievedChunk], str]:
    """Retrieve Ask-mode candidates and rerank them when possible."""

    pinecone_chunks = retrieve_chunks(question, top_k=ASK_RETRIEVAL_CANDIDATE_TOP_K, namespace=namespace)
    if not pinecone_chunks:
        return [], "corpus"

    try:
        reranked_chunks = rerank_chunks(
            question,
            pinecone_chunks,
            top_n=min(ASK_FINAL_TOP_K, len(pinecone_chunks)),
        )
    except RerankingError:
        logger.exception(
            "Cohere reranking failed for Ask-mode question %r; falling back to Pinecone top %d",
            question,
            ASK_FINAL_TOP_K,
        )
        return pinecone_chunks[:ASK_FINAL_TOP_K], "corpus"

    selected_chunks = [
        _reranked_to_retrieved_chunk(chunk)
        for chunk in reranked_chunks[:ASK_FINAL_TOP_K]
    ]
    return selected_chunks, "reranked"


def _rerank_candidate_chunks(
    question: str,
    candidate_chunks: Sequence[RetrievedChunk],
    *,
    scope: str,
    final_top_k: int = LEARN_FINAL_TOP_K,
) -> tuple[list[RetrievedChunk], str]:
    """Rerank candidate chunks and fall back to the original Pinecone order if needed."""

    if not candidate_chunks:
        return [], scope

    try:
        reranked_chunks = rerank_chunks(
            question,
            candidate_chunks,
            top_n=min(final_top_k, len(candidate_chunks)),
        )
    except RerankingError:
        logger.exception(
            "Cohere reranking failed for Learn-mode question %r in %s scope; falling back to Pinecone top %d",
            question,
            scope,
            final_top_k,
        )
        return list(candidate_chunks[:final_top_k]), scope

    selected_chunks = [
        _reranked_to_retrieved_chunk(chunk)
        for chunk in reranked_chunks[:final_top_k]
    ]
    return selected_chunks, scope


def _unique_citations(chunks: Sequence[RetrievedChunk]) -> list[SourceCitation]:
    """Build deterministic unique citations from retrieved chunks."""

    citations: list[SourceCitation] = []
    seen_titles: set[str] = set()
    for chunk in chunks:
        citation = SourceCitation.from_chunk(chunk)
        title = str(citation.document_title or "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        citations.append(citation)
    return citations


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a strict JSON object from model output."""

    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("empty response")

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    return payload


def _lesson_response_format() -> dict[str, Any]:
    """Build the structured-output request for lesson generation."""

    return {
        "type": "json_schema",
        "name": "lesson_generation",
        "schema": LESSON_RESPONSE_SCHEMA,
        "strict": True,
    }


def _retrieve_grounded_chunks(
    question: str,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
    document_paths: Sequence[str] | None = None,
    allow_fallback: bool = False,
) -> tuple[list[RetrievedChunk], str]:
    """Retrieve chunks with optional strict document scoping."""

    if document_paths is None:
        chunks = retrieve_chunks(question, top_k=top_k, namespace=namespace)
        return _rerank_candidate_chunks(question, chunks, scope="corpus")

    strict_chunks = retrieve_chunks(
        question,
        top_k=top_k,
        namespace=namespace,
        document_paths=document_paths,
    )
    if strict_chunks:
        return _rerank_candidate_chunks(question, strict_chunks, scope="module")

    if allow_fallback:
        logger.info(
            "Strict module retrieval returned no usable context for %r; falling back to broad retrieval",
            question,
        )
        fallback_chunks = retrieve_chunks(question, top_k=top_k, namespace=namespace)
        return _rerank_candidate_chunks(question, fallback_chunks, scope="fallback")

    return [], "module"


def _generate_answer_state(
    question: str,
    *,
    top_k: int = ASK_FINAL_TOP_K,
    namespace: str | None = None,
    document_paths: Sequence[str] | None = None,
    allow_fallback: bool = False,
) -> AskState:
    """Retrieve context, generate an answer, and attach citations."""

    state = AskState(
        question=question,
        top_k=top_k,
        namespace=namespace,
        document_paths=list(document_paths) if document_paths is not None else None,
        allow_fallback=allow_fallback,
    )
    if document_paths is None:
        state.retrieved_chunks, state.retrieval_scope = _select_ask_chunks(
            question,
            namespace=namespace,
        )
    else:
        state.retrieved_chunks, state.retrieval_scope = _retrieve_grounded_chunks(
            question,
            top_k=top_k,
            namespace=namespace,
            document_paths=document_paths,
            allow_fallback=allow_fallback,
        )
    state.context = _build_context(state.retrieved_chunks)
    state = _generate(state)
    state = _respond(state)
    return state


def _lesson_prompt(module: CurriculumModule, context: str) -> str:
    """Construct the lesson-generation prompt."""

    prompt_template = _get_lesson_prompt_template()
    module_objectives = "\n".join(f"- {objective}" for objective in module.objectives)
    return prompt_template.format(
        module_title=module.title,
        module_description=module.description,
        module_objectives=module_objectives,
        context=context,
    )


def _build_safe_lesson(
    module: CurriculumModule,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    retrieval_scope: str,
) -> LearningLesson:
    """Return a safe lesson placeholder when context is unavailable."""

    return LearningLesson(
        module_id=module.id,
        module_title=module.title,
        lesson_title=module.title,
        learning_content=LESSON_INSUFFICIENT_INFORMATION,
        key_takeaways=[],
        citations=_unique_citations(retrieved_chunks),
        retrieved_chunks=list(retrieved_chunks),
        retrieval_scope=retrieval_scope,
    )


def _extract_module_id(raw_text: str, allowed_modules: Sequence[CurriculumModule]) -> str | None:
    """Extract a curriculum module id from the LLM output."""

    allowed_ids = {module.id for module in allowed_modules}
    title_to_id = {module.title.strip().lower(): module.id for module in allowed_modules}
    cleaned = raw_text.strip()

    if not cleaned:
        return None

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = cleaned

    candidate: str | None = None
    if isinstance(payload, dict):
        value = payload.get("module_id") or payload.get("id") or payload.get("module")
        if isinstance(value, str):
            candidate = value.strip()
    elif isinstance(payload, str):
        candidate = payload.strip()

    if candidate in allowed_ids:
        return candidate

    if candidate and candidate.lower() in title_to_id:
        return title_to_id[candidate.lower()]

    if cleaned in allowed_ids:
        return cleaned

    if cleaned.lower() in title_to_id:
        return title_to_id[cleaned.lower()]

    return None


def _build_starting_module_prompt(profile: LearnerProfile, allowed_modules: Sequence[CurriculumModule]) -> str:
    """Construct the onboarding prompt used to recommend a starting module."""

    allowed_lines = "\n".join(
        [
            f"- {module.id}: {module.title} | difficulty={module.difficulty} | duration={module.duration}"
            for module in allowed_modules
        ]
    )

    return "\n".join(
        [
            "You are selecting a starting curriculum module for a learner.",
            "Return only a valid module id from the allowed list.",
            "Choose the best starting point for this learner profile.",
            f"Background: {profile.academic_or_professional_background}",
            f"Drug development familiarity: {profile.drug_development_familiarity}",
            f"Learning goal: {profile.learning_goal}",
            f"Prior regulatory/pharma experience: {profile.prior_regulatory_pharma_experience}",
            f"Available study time: {profile.available_study_time}",
            "Allowed modules:",
            allowed_lines,
        ]
    )


def recommend_starting_module_id(
    profile: LearnerProfile,
) -> str:
    """Use the LLM to choose a valid starting module id."""

    allowed_modules = curriculum_next_modules([])
    if not allowed_modules:
        allowed_modules = [earliest_entry_module()]

    prompt = _build_starting_module_prompt(profile, allowed_modules)

    try:
        response = client.responses.create(model=CHAT_MODEL, input=prompt)
    except Exception as exc:  # pragma: no cover - exercised in failure tests
        logger.exception("Starting-module recommendation failed")
        return allowed_modules[0].id

    candidate = _extract_module_id(getattr(response, "output_text", ""), allowed_modules)
    if candidate is None:
        logger.info(
            "LLM returned an invalid starting module recommendation; falling back to %s",
            allowed_modules[0].id,
        )
        return allowed_modules[0].id
    return candidate


def start_learning_session(profile: LearnerProfile) -> LearningSession:
    """Create a new in-memory learning session."""

    recommended_start_module_id = recommend_starting_module_id(profile)
    return LearningSession(
        profile=profile,
        recommended_start_module_id=recommended_start_module_id,
        current_module_id=recommended_start_module_id,
    )


def _module_for_id(module_id: str) -> CurriculumModule:
    module = get_module(module_id)
    if module is None:
        raise ValueError(f"Unknown curriculum module: {module_id}")
    return module


def answer_learning_question(
    module_id: str,
    question: str,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
) -> LearningAnswer:
    """Answer a learner question using module-scoped retrieval."""

    module = _module_for_id(module_id)
    state = _generate_answer_state(
        question,
        top_k=top_k,
        namespace=namespace,
        document_paths=module.documents,
        allow_fallback=True,
    )
    return LearningAnswer(
        module_id=module.id,
        module_title=module.title,
        question=question,
        answer=state.answer,
        citations=state.citations,
        retrieved_chunks=state.retrieved_chunks,
        retrieval_scope=state.retrieval_scope,
    )


def explain_learning_module(
    module_id: str,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
) -> LearningAnswer:
    """Generate a grounded explanation for the current curriculum module."""

    module = _module_for_id(module_id)
    objective_summary = "; ".join(module.objectives)
    explanation_question = (
        f"Explain the curriculum module '{module.title}'. "
        f"Focus on the module description and these learning objectives: {objective_summary}. "
        "Use only the provided documents and keep the explanation concise but useful."
    )
    return answer_learning_question(
        module.id,
        explanation_question,
        top_k=top_k,
        namespace=namespace,
    )


def generate_learning_lesson(
    module_id: str,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
) -> LearningLesson:
    """Generate a grounded lesson for a curriculum module."""

    module = _module_for_id(module_id)
    lesson_query = " ".join(
        [
            module.title,
            module.description,
            "; ".join(module.objectives),
        ]
    )
    retrieved_chunks, retrieval_scope = _retrieve_grounded_chunks(
        lesson_query,
        top_k=top_k,
        namespace=namespace,
        document_paths=module.documents,
        allow_fallback=False,
    )
    usable_chunks = [chunk for chunk in retrieved_chunks if chunk.text.strip()]
    if not usable_chunks:
        logger.info("No usable context available for lesson generation for module %s", module.id)
        return _build_safe_lesson(module, retrieved_chunks, retrieval_scope=retrieval_scope)

    context = _build_context(usable_chunks)
    if not context.strip():
        logger.info("Lesson context was empty for module %s", module.id)
        return _build_safe_lesson(module, usable_chunks, retrieval_scope=retrieval_scope)

    prompt = _lesson_prompt(module, context)

    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
            text={"format": _lesson_response_format()},
        )
    except Exception as exc:  # pragma: no cover - exercised via failure tests
        logger.exception("Lesson generation failed")
        raise LessonGenerationError("Failed to generate a grounded lesson") from exc

    output_text = getattr(response, "output_text", "")
    try:
        payload = _extract_json_object(output_text)
    except Exception as exc:  # pragma: no cover - guarded by tests
        logger.exception("Lesson generation returned malformed JSON")
        raise LessonGenerationError("Lesson generation must return strict JSON") from exc

    lesson_title = str(payload.get("lesson_title", "")).strip()
    learning_content = str(payload.get("learning_content", "")).strip()
    raw_takeaways = payload.get("key_takeaways", [])

    if not lesson_title or not learning_content:
        raise LessonGenerationError("Lesson generation returned incomplete content")
    if not isinstance(raw_takeaways, list):
        raise LessonGenerationError("Lesson key takeaways must be returned as a list")

    key_takeaways = [str(item).strip() for item in raw_takeaways if str(item).strip()]
    if not key_takeaways:
        raise LessonGenerationError("Lesson generation returned no key takeaways")

    citations = _unique_citations(usable_chunks)
    logger.info("Generated lesson for module %s from %d chunks", module.id, len(usable_chunks))
    return LearningLesson(
        module_id=module.id,
        module_title=module.title,
        lesson_title=lesson_title,
        learning_content=learning_content,
        key_takeaways=key_takeaways,
        citations=citations,
        retrieved_chunks=list(usable_chunks),
        retrieval_scope=retrieval_scope,
    )


def ensure_learning_lesson(
    session: LearningSession,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
    force: bool = False,
) -> LearningLesson:
    """Return the cached lesson for the active module or generate a new one."""

    current_lesson = getattr(session, "current_lesson", None)
    if (
        not force
        and current_lesson is not None
        and getattr(current_lesson, "module_id", None) == session.current_module_id
    ):
        return current_lesson

    lesson = generate_learning_lesson(
        session.current_module_id,
        top_k=top_k,
        namespace=namespace,
    )
    session.current_lesson = lesson
    return lesson


def generate_learning_quiz(
    module_id: str,
    *,
    top_k: int = LEARN_RETRIEVAL_CANDIDATE_TOP_K,
    namespace: str | None = None,
    lesson: LearningLesson | None = None,
) -> LearningQuizBundle:
    """Generate a short quiz for a curriculum module."""

    module = _module_for_id(module_id)
    module_query = " ".join(
        [
            module.title,
            module.description,
            "; ".join(module.objectives),
        ]
    )
    retrieved_chunks, retrieval_scope = _retrieve_grounded_chunks(
        module_query,
        top_k=top_k,
        namespace=namespace,
        document_paths=module.documents,
        allow_fallback=True,
    )
    quiz = generate_quiz(
        module,
        retrieved_chunks,
        lesson_title=lesson.lesson_title if lesson is not None else None,
        lesson_content=lesson.learning_content if lesson is not None else None,
        lesson_takeaways=lesson.key_takeaways if lesson is not None else None,
    )
    return LearningQuizBundle(
        module_id=module.id,
        module_title=module.title,
        quiz=quiz,
        retrieved_chunks=list(retrieved_chunks),
        retrieval_scope=retrieval_scope,
    )


def evaluate_learning_quiz(
    quiz_bundle: LearningQuizBundle,
    lesson_content: str,
    learner_answers: Sequence[str],
) -> QuizEvaluation:
    """Grade learner answers against the retrieved curriculum context."""

    return evaluate_quiz(quiz_bundle.quiz, learner_answers, lesson_content, quiz_bundle.retrieved_chunks)


def recommend_next_module(
    completed_module_ids: Sequence[str],
) -> CurriculumModule | None:
    """Recommend the next valid curriculum module after completion."""

    next_modules = curriculum_next_modules(completed_module_ids)
    return next_modules[0] if next_modules else None


def preview_next_module(session: LearningSession) -> CurriculumModule | None:
    """Preview the next valid module after the current module is completed."""

    completed = list(session.completed_module_ids)
    if session.current_module_id not in completed:
        completed.append(session.current_module_id)
    return recommend_next_module(completed)


def complete_current_module(session: LearningSession) -> CurriculumModule | None:
    """Mark the current module as completed and advance to the next one if available."""

    if session.current_module_id not in session.completed_module_ids:
        session.completed_module_ids.append(session.current_module_id)

    next_module = recommend_next_module(session.completed_module_ids)
    session.quiz_result = None
    session.current_quiz = None
    session.current_lesson = None

    if next_module is not None:
        session.current_module_id = next_module.id
    return next_module


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

        state = _generate_answer_state(
            question,
            top_k=top_k,
            namespace=namespace,
            allow_fallback=False,
        )
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

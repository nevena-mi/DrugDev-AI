"""Streamlit application for the Ask / Learn / Monitor interface."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from src.chatbot import ask_question
from src.curriculum import get_module, list_modules, prerequisites_satisfied
from src.graph import (
    LearningAnswer,
    LearningSession,
    LearnerProfile,
    answer_learning_question,
    complete_current_module,
    ensure_learning_lesson,
    evaluate_learning_quiz,
    generate_learning_quiz,
    preview_next_module,
    start_learning_session,
)


logger = logging.getLogger(__name__)


class StreamlitLike(Protocol):
    """Minimal Streamlit surface used by the application."""

    session_state: Any

    def set_page_config(self, **kwargs: Any) -> None: ...
    def title(self, text: str) -> None: ...
    def markdown(self, text: str) -> None: ...
    def write(self, value: Any) -> None: ...
    def subheader(self, text: str) -> None: ...
    def text_input(self, label: str, value: str = "", placeholder: str = "") -> str: ...
    def text_area(self, label: str, value: str = "", placeholder: str = "") -> str: ...
    def button(self, label: str) -> bool: ...
    def error(self, text: str) -> None: ...
    def success(self, text: str) -> None: ...
    def info(self, text: str) -> None: ...
    def warning(self, text: str) -> None: ...
    def divider(self) -> None: ...
    def tabs(self, labels: list[str]) -> list[Any]: ...
    def container(self) -> Any: ...


def _get_streamlit(st: StreamlitLike | None = None) -> StreamlitLike:
    """Return the provided Streamlit adapter or import the real library."""

    if st is not None:
        return st

    try:
        import streamlit as real_streamlit
    except ImportError as exc:  # pragma: no cover - exercised in manual runtime
        raise RuntimeError(
            "Streamlit is required to run the application. Install it and run `streamlit run src/app.py`."
        ) from exc

    return real_streamlit


def _get_session_state(st: StreamlitLike) -> dict[str, Any]:
    """Return the mutable Streamlit session state mapping."""

    state = getattr(st, "session_state", None)
    if state is None:
        raise RuntimeError("Streamlit session_state is not available")
    return state


def _render_placeholder(st: StreamlitLike, title: str) -> None:
    """Render a simple placeholder panel for future modes."""

    st.title(title)
    st.info("This mode will be implemented in a later phase.")


def _render_citation_block(st: StreamlitLike, citations: list[Any]) -> None:
    """Render unique document titles for the current answer."""

    if not citations:
        st.info("No citations available for this answer.")
        return

    st.subheader("Citations")
    seen_titles: set[str] = set()
    rendered_titles: list[str] = []

    for citation in citations:
        citation_data = _to_mapping(citation)
        title = str(citation_data.get("document_title", "")).strip()
        if not title or title in seen_titles:
            continue

        seen_titles.add(title)
        rendered_titles.append(title)
        st.markdown(f"**Citation {len(rendered_titles)}**: {title}")


def _render_document_titles(st: StreamlitLike, citations: list[Any]) -> None:
    """Render unique document titles for a lesson."""

    if not citations:
        st.info("No citations available for this lesson.")
        return

    st.subheader("Citations")
    seen_titles: set[str] = set()
    for citation in citations:
        citation_data = _to_mapping(citation)
        title = str(citation_data.get("document_title", "")).strip()
        if not title or title in seen_titles:
            continue

        seen_titles.add(title)
        st.markdown(f"- {title}")


def _render_answer_block(st: StreamlitLike, heading: str, answer: LearningAnswer | None) -> None:
    """Render a grounded answer with citations."""

    if answer is None:
        return

    st.subheader(heading)
    st.write(answer.answer)
    if answer.retrieval_scope == "fallback":
        st.info("Used a broader retrieval fallback because the module documents did not yield context.")
    _render_citation_block(st, list(answer.citations))


def _to_mapping(value: Any) -> dict[str, Any]:
    """Convert a citation-like object into a plain mapping."""

    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "items"):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": value}


def _render_ask_tab(st: StreamlitLike, ask_fn=ask_question) -> None:
    """Render the functional Ask tab."""

    st.title("Ask")
    st.markdown(
        "Ask a natural-language question and receive an answer grounded in the indexed regulatory documents."
    )

    question = st.text_input(
        "Your question",
        value="",
        placeholder="e.g. What is Good Clinical Practice?",
    )

    if not st.button("Ask"):
        return

    if not question.strip():
        st.warning("Enter a question before pressing Ask.")
        return

    try:
        result = ask_fn(question.strip())
    except Exception as exc:  # pragma: no cover - exercised in manual runtime
        logger.exception("Ask pipeline failed")
        st.error(f"Unable to answer the question: {exc}")
        return

    st.subheader("Answer")
    st.write(result.answer)
    _render_citation_block(st, list(result.citations))


def _render_onboarding_form(st: StreamlitLike, state: dict[str, Any]) -> None:
    """Collect learner onboarding information and start a learning session."""

    st.subheader("Onboarding")
    st.markdown("Tell us a little about your background so we can suggest a valid starting module.")

    background = st.text_input(
        "Academic or professional background",
        value=state.get("learn_background", ""),
    )
    familiarity = st.text_input(
        "Familiarity with drug development",
        value=state.get("learn_familiarity", ""),
    )
    goal = st.text_input(
        "Learning goal",
        value=state.get("learn_goal", ""),
    )
    experience = st.text_input(
        "Prior regulatory or pharma experience",
        value=state.get("learn_experience", ""),
    )
    study_time = st.text_input(
        "Available study time",
        value=state.get("learn_study_time", ""),
    )

    state["learn_background"] = background
    state["learn_familiarity"] = familiarity
    state["learn_goal"] = goal
    state["learn_experience"] = experience
    state["learn_study_time"] = study_time

    if not st.button("Start learning"):
        return

    profile = LearnerProfile(
        academic_or_professional_background=background.strip(),
        drug_development_familiarity=familiarity.strip(),
        learning_goal=goal.strip(),
        prior_regulatory_pharma_experience=experience.strip(),
        available_study_time=study_time.strip(),
    )
    session = start_learning_session(profile)
    state["learning_session"] = session
    state.pop("learning_module_answer", None)
    state.pop("learning_quiz_bundle", None)
    state.pop("learning_quiz_result", None)
    st.success(f"Starting module selected: {session.current_module_id}")


def _render_curriculum_overview(st: StreamlitLike, session: LearningSession) -> None:
    """Render the curriculum and the learner's progression state."""

    st.subheader("Curriculum Overview")
    completed = set(session.completed_module_ids)
    for module in list_modules():
        if module.id == session.current_module_id:
            status = "Current"
        elif module.id in completed:
            status = "Completed"
        elif prerequisites_satisfied(module.id, session.completed_module_ids):
            status = "Available"
        else:
            status = "Locked"
        st.markdown(f"- **{module.title}** (`{module.id}`) - {status}")


def _render_current_module_panel(st: StreamlitLike, session: LearningSession) -> None:
    """Show the active curriculum module and the next valid recommendation."""

    module = get_module(session.current_module_id)
    if module is None:
        st.error(f"Unknown curriculum module: {session.current_module_id}")
        return

    st.subheader("Current Module")
    st.markdown(f"**{module.title}**")
    st.markdown(module.description)
    st.markdown("**Objectives**")
    for objective in module.objectives:
        st.markdown(f"- {objective}")

    st.info(f"Recommended starting module: {session.recommended_start_module_id}")


def _render_learning_lesson(st: StreamlitLike, session: LearningSession) -> None:
    """Render the generated lesson for the active curriculum module."""

    try:
        lesson = ensure_learning_lesson(session)
    except Exception as exc:  # pragma: no cover - exercised in manual runtime
        logger.exception("Learn mode lesson generation failed")
        st.error(f"Unable to generate the lesson: {exc}")
        return

    st.subheader("Learning Content")
    st.markdown(f"**{lesson.lesson_title}**")
    st.write(lesson.learning_content)

    st.subheader("Key Takeaways")
    if lesson.key_takeaways:
        for takeaway in lesson.key_takeaways:
            st.markdown(f"- {takeaway}")
    else:
        st.info("No key takeaways were returned for this lesson.")

    _render_document_titles(st, list(lesson.citations))


def _render_module_question_box(st: StreamlitLike, session: LearningSession, state: dict[str, Any]) -> None:
    """Let the learner ask a free-form question about the current module."""

    st.subheader("Ask About This Module")
    question = st.text_input(
        "Your learning question",
        value=state.get("learn_question", ""),
        placeholder="e.g. How does this module connect to clinical trial design?",
    )
    state["learn_question"] = question

    if st.button("Ask module question"):
        if not question.strip():
            st.warning("Enter a question before asking about the module.")
            return
        try:
            state["learning_question_answer"] = answer_learning_question(
                session.current_module_id,
                question.strip(),
            )
        except Exception as exc:  # pragma: no cover - exercised in manual runtime
            logger.exception("Learn mode question failed")
            st.error(f"Unable to answer the learning question: {exc}")

    _render_answer_block(st, "Question Answer", state.get("learning_question_answer"))


def _render_quiz_section(st: StreamlitLike, session: LearningSession, state: dict[str, Any]) -> None:
    """Render quiz generation, answering and evaluation."""

    st.subheader("Quiz")

    if st.button("Generate quiz"):
        try:
            session.current_quiz = generate_learning_quiz(session.current_module_id)
            session.quiz_result = None
            state["learning_quiz_result"] = None
        except Exception as exc:  # pragma: no cover - exercised in manual runtime
            logger.exception("Learn mode quiz generation failed")
            st.error(f"Unable to generate a quiz: {exc}")

    quiz_bundle = session.current_quiz
    if quiz_bundle is None:
        st.info("Generate a quiz to check your understanding of the current module.")
        return

    st.markdown(f"Quiz grounded in: {'; '.join(quiz_bundle.quiz.source_document_titles) or 'module documents'}")
    answers: list[str] = []
    for question in quiz_bundle.quiz.questions:
        answer = st.text_area(
            f"{question.id}: {question.question}",
            value=state.get(f"quiz_answer_{question.id}", ""),
        )
        state[f"quiz_answer_{question.id}"] = answer
        answers.append(answer)

    if st.button("Submit quiz"):
        try:
            evaluation = evaluate_learning_quiz(quiz_bundle, answers)
            session.quiz_result = evaluation
            state["learning_quiz_result"] = evaluation
            st.success("Quiz submitted.")
        except Exception as exc:  # pragma: no cover - exercised in manual runtime
            logger.exception("Learn mode quiz evaluation failed")
            st.error(f"Unable to evaluate the quiz: {exc}")

    if session.quiz_result is not None:
        st.subheader("Quiz Result")
        st.markdown(f"Score: {session.quiz_result.score:.2f}")
        st.markdown("Passed" if session.quiz_result.passed else "Not yet passed")
        if session.quiz_result.feedback:
            st.markdown(session.quiz_result.feedback)
        for feedback in session.quiz_result.question_feedback:
            st.markdown(f"- {feedback}")

    if session.quiz_result is not None and session.quiz_result.passed:
        if st.button("Complete current module"):
            next_module = complete_current_module(session)
            state.pop("learning_question_answer", None)
            state["learning_quiz_result"] = None
            for key in list(state):
                if key.startswith("quiz_answer_"):
                    state.pop(key, None)
            if next_module is not None:
                st.success(f"Advanced to {next_module.title}")
            else:
                st.success("You have completed the current curriculum path.")


def _render_progress_section(st: StreamlitLike, session: LearningSession) -> None:
    """Render learner progress and completion state."""

    st.subheader("Progress")
    completed_modules = [
        module.title
        for module in list_modules()
        if module.id in session.completed_module_ids
    ]
    if completed_modules:
        st.markdown("**Completed modules**")
        for module_title in completed_modules:
            st.markdown(f"- {module_title}")
    else:
        st.info("No modules completed yet.")

    next_module = preview_next_module(session)
    if next_module is not None:
        st.info(f"Next recommended module: {next_module.title}")


def _render_learn_tab(st: StreamlitLike) -> None:
    """Render the functional Learn tab."""

    state = _get_session_state(st)
    session = state.get("learning_session")

    st.title("Learn")
    st.markdown(
        "Follow the curriculum, study the current module, ask questions, and check your understanding."
    )

    if session is None:
        _render_onboarding_form(st, state)
        session = state.get("learning_session")
        if session is None:
            return

    _render_curriculum_overview(st, session)
    _render_current_module_panel(st, session)
    _render_learning_lesson(st, session)
    _render_module_question_box(st, session, state)
    _render_quiz_section(st, session, state)
    _render_progress_section(st, session)


def main(st: StreamlitLike | None = None, *, ask_fn=ask_question) -> None:
    """Run the Streamlit application."""

    ui = _get_streamlit(st)
    ui.set_page_config(page_title="DrugDev AI", page_icon="📚", layout="wide")

    ui.title("DrugDev AI")
    ui.markdown("A regulatory science assistant for grounded Ask-mode question answering.")

    ask_tab, learn_tab, monitor_tab = ui.tabs(["Ask", "Learn", "Monitor"])

    with ask_tab:
        _render_ask_tab(ui, ask_fn=ask_fn)

    with learn_tab:
        _render_learn_tab(ui)

    with monitor_tab:
        _render_placeholder(ui, "Monitor")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()

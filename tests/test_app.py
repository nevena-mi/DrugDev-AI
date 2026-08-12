from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import src.app as app_module
import src.monitor as monitor_module


@dataclass
class FakeTab:
    label: str

    def __enter__(self) -> "FakeTab":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeStreamlit:
    def __init__(
        self,
        *,
        question: str = "",
        text_inputs: dict[str, str] | None = None,
        text_areas: dict[str, str] | None = None,
        checkboxes: dict[str, bool] | None = None,
        buttons: dict[str, bool] | None = None,
        session_state: dict[str, object] | None = None,
    ) -> None:
        self.question = question
        self.text_inputs = text_inputs or {}
        self.text_areas = text_areas or {}
        self.checkboxes = checkboxes or {}
        self.buttons = buttons or {}
        self.session_state = session_state or {}
        self.calls: list[tuple[str, object]] = []
        self.rerun_called = False

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, text: str) -> None:
        self.calls.append(("title", text))

    def markdown(self, text: str, **kwargs) -> None:
        self.calls.append(("markdown", text))

    def write(self, value) -> None:
        self.calls.append(("write", value))

    def subheader(self, text: str) -> None:
        self.calls.append(("subheader", text))

    def text_input(self, label: str, value: str = "", placeholder: str = "") -> str:
        self.calls.append(("text_input", (label, value, placeholder)))
        if label == "Your question" and self.question:
            return self.question
        return self.text_inputs.get(label, value)

    def text_area(self, label: str, value: str = "", placeholder: str = "") -> str:
        self.calls.append(("text_area", (label, value, placeholder)))
        return self.text_areas.get(label, value)

    def checkbox(self, label: str, value: bool = False) -> bool:
        self.calls.append(("checkbox", (label, value)))
        return self.checkboxes.get(label, value)

    def button(self, label: str, key: str | None = None) -> bool:
        self.calls.append(("button", (label, key)))
        lookup_key = key if key is not None else label
        return self.buttons.get(lookup_key, False)

    def error(self, text: str) -> None:
        self.calls.append(("error", text))

    def success(self, text: str) -> None:
        self.calls.append(("success", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def rerun(self) -> None:
        self.rerun_called = True
        self.calls.append(("rerun", None))

    def divider(self) -> None:
        self.calls.append(("divider", None))

    def tabs(self, labels: list[str]) -> list[FakeTab]:
        self.calls.append(("tabs", tuple(labels)))
        return [FakeTab(label) for label in labels]


def test_app_imports_cleanly() -> None:
    assert hasattr(app_module, "main")


def test_main_applies_shared_content_width_shell() -> None:
    fake_streamlit = FakeStreamlit(
        question="What is GCP?",
        buttons={"Ask": True},
    )

    app_module.main(fake_streamlit, ask_fn=lambda question: SimpleNamespace(answer="ok", citations=[]))

    assert any(
        call[0] == "markdown" and "max-width: 980px" in call[1] and ".block-container" in call[1]
        for call in fake_streamlit.calls
    )


def test_main_renders_ask_and_title_only_citations() -> None:
    fake_result = SimpleNamespace(
        answer="Grounded answer",
        citations=[
            SimpleNamespace(document_title="example"),
            SimpleNamespace(document_title="example"),
            SimpleNamespace(document_title="other"),
        ],
    )
    fake_streamlit = FakeStreamlit(
        question="What is GCP?",
        buttons={"Ask": True},
    )

    app_module.main(fake_streamlit, ask_fn=lambda question: fake_result)

    assert ("tabs", ("Ask", "Learn", "Monitor")) in fake_streamlit.calls
    assert ("subheader", "Answer") in fake_streamlit.calls
    assert ("write", "Grounded answer") in fake_streamlit.calls
    assert any(call[0] == "markdown" and "**Citation 1**: example" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "**Citation 2**: other" in call[1] for call in fake_streamlit.calls)
    assert not any(
        call[0] == "markdown"
        and any(token in call[1] for token in ["ID:", "File:", "Organization:", "Chunk ID:", "Similarity score:"])
        for call in fake_streamlit.calls
    )


def test_main_starts_learning_session_from_onboarding() -> None:
    profile_capture: dict[str, object] = {}
    fake_session = SimpleNamespace(
        profile=SimpleNamespace(),
        recommended_start_module_id="foundations",
        current_module_id="foundations",
        completed_module_ids=[],
        quiz_result=None,
        current_quiz=None,
        current_lesson=None,
    )
    modules = [
        SimpleNamespace(
            id="foundations",
            title="Introduction to Drug Development",
            description="Overview of the pharmaceutical industry.",
            prerequisites=[],
            objectives=["Understand the drug development lifecycle"],
        ),
        SimpleNamespace(
            id="regulatory",
            title="Regulatory Landscape",
            description="Introduction to EMA, FDA, ICH and WHO.",
            prerequisites=["foundations"],
            objectives=["Understand the role of regulatory agencies"],
        ),
    ]

    fake_streamlit = FakeStreamlit(
        text_inputs={
            "Academic or professional background": "Scientist",
            "Familiarity with drug development": "Beginner",
            "Learning goal": "Learn the basics",
            "Prior regulatory or pharma experience": "None",
            "Available study time": "3 hours/week",
        },
        buttons={"Start learning": True},
    )

    def fake_start_learning_session(profile):
        profile_capture["profile"] = profile
        return fake_session

    with (
        patch.object(app_module, "start_learning_session", side_effect=fake_start_learning_session),
        patch.object(app_module, "list_modules", return_value=modules),
        patch.object(app_module, "get_module", return_value=modules[0]),
        patch.object(app_module, "prerequisites_satisfied", side_effect=lambda module_id, completed: module_id == "foundations"),
        patch.object(app_module, "preview_next_module", return_value=modules[1]),
    ):
        app_module.main(fake_streamlit)

    assert profile_capture["profile"].learning_goal == "Learn the basics"
    assert fake_streamlit.session_state["learning_session"] is fake_session
    assert ("subheader", "Curriculum Overview") in fake_streamlit.calls
    assert ("subheader", "Current Module") in fake_streamlit.calls
    assert any(call[0] == "success" and "Starting module selected" in call[1] for call in fake_streamlit.calls)


def test_main_routes_learn_interactions_to_backend() -> None:
    fake_session = SimpleNamespace(
        profile=SimpleNamespace(),
        recommended_start_module_id="foundations",
        current_module_id="foundations",
        completed_module_ids=[],
        quiz_result=None,
        current_quiz=None,
        current_lesson=None,
    )
    fake_quiz_bundle = SimpleNamespace(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        quiz=SimpleNamespace(
            source_document_titles=["EMA Human Regulatory Overview"],
            questions=[
                SimpleNamespace(id="q1", question="What is GCP?"),
                SimpleNamespace(id="q2", question="Name one stakeholder."),
            ],
        ),
        retrieved_chunks=[SimpleNamespace(text="context", metadata={})],
        retrieval_scope="module",
    )
    fake_lesson = SimpleNamespace(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        lesson_title="Drug Development Foundations",
        learning_content="Grounded lesson content.",
        key_takeaways=["Takeaway one", "Takeaway two"],
        citations=[
            SimpleNamespace(document_title="EMA Human Regulatory Overview"),
            SimpleNamespace(document_title="EMA Human Regulatory Overview"),
            SimpleNamespace(document_title="FDA Drug Development and Approval Process"),
        ],
        retrieved_chunks=[SimpleNamespace(text="context", metadata={})],
        retrieval_scope="module",
    )
    fake_quiz_result = SimpleNamespace(
        module_id="foundations",
        number_correct=2,
        total_questions=5,
        percentage=40.0,
        passed=False,
        question_feedback=[
            SimpleNamespace(id="q1", correct=True, explanation="Good Clinical Practice is a standard for trials."),
            SimpleNamespace(id="q2", correct=False, explanation="The answer missed the application."),
        ],
    )
    fake_answer = SimpleNamespace(
        module_id="foundations",
        module_title="Introduction to Drug Development",
        question="How does this connect to clinical trials?",
        answer="Grounded answer",
        citations=[SimpleNamespace(document_title="example")],
        retrieved_chunks=[],
        retrieval_scope="module",
    )

    fake_streamlit = FakeStreamlit(
        session_state={"learning_session": fake_session},
        text_inputs={
            "Your learning question": "How does this connect to clinical trials?",
        },
        text_areas={
            "q1: What is GCP?": "Good Clinical Practice",
            "q2: Name one stakeholder.": "Sponsors",
        },
        buttons={
            "Ask module question": True,
            "Generate quiz": True,
            "Submit quiz": True,
            "Complete current module": True,
        },
    )

    with (
        patch.object(app_module, "answer_learning_question", return_value=fake_answer) as answer_question,
        patch.object(app_module, "ensure_learning_lesson", return_value=fake_lesson) as ensure_lesson,
        patch.object(app_module, "generate_learning_quiz", return_value=fake_quiz_bundle) as generate_quiz,
        patch.object(app_module, "evaluate_learning_quiz", return_value=fake_quiz_result) as evaluate_quiz,
        patch.object(app_module, "complete_current_module", return_value=SimpleNamespace(title="Regulatory Landscape")) as complete_module,
        patch.object(app_module, "list_modules", return_value=[]),
        patch.object(app_module, "get_module", return_value=SimpleNamespace(
            id="foundations",
            title="Introduction to Drug Development",
            description="Overview of the pharmaceutical industry.",
            objectives=["Understand the drug development lifecycle"],
        )),
        patch.object(app_module, "prerequisites_satisfied", return_value=True),
        patch.object(app_module, "preview_next_module", return_value=SimpleNamespace(title="Regulatory Landscape")),
    ):
        app_module.main(fake_streamlit, ask_fn=lambda question: SimpleNamespace(answer="", citations=[], retrieved_chunks=[]))

    assert answer_question.call_count == 1
    assert ensure_lesson.call_count == 1
    assert generate_quiz.call_count == 1
    assert evaluate_quiz.call_count == 1
    assert complete_module.call_count == 0
    assert evaluate_quiz.call_args.args[1] == "Grounded lesson content."
    assert evaluate_quiz.call_args.args[2] == ["Good Clinical Practice", "Sponsors"]
    assert fake_session.quiz_result is fake_quiz_result
    assert any(call[0] == "subheader" and call[1] == "Learning Content" for call in fake_streamlit.calls)
    assert any(call[0] == "write" and call[1] == "Grounded lesson content." for call in fake_streamlit.calls)
    assert any(call[0] == "subheader" and call[1] == "Key Takeaways" for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "- Takeaway one" in call[1] for call in fake_streamlit.calls)
    assert sum(
        1
        for call in fake_streamlit.calls
        if call[0] == "markdown" and "- EMA Human Regulatory Overview" in call[1]
    ) == 1
    assert any(call[0] == "markdown" and "- FDA Drug Development and Approval Process" in call[1] for call in fake_streamlit.calls)
    assert not any("Retrieved Sources" in str(call[1]) for call in fake_streamlit.calls if call[0] == "subheader")
    assert any(call[0] == "subheader" and call[1] == "Question Answer" for call in fake_streamlit.calls)
    assert any(call[0] == "subheader" and call[1] == "Quiz Result" for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "Score: 2/5 (40%)" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "Needs review" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "**Q1 — Correct**" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "Good Clinical Practice is a standard for trials." in call[1] for call in fake_streamlit.calls)


def test_main_handles_backend_errors_gracefully() -> None:
    fake_streamlit = FakeStreamlit(question="What is GCP?", buttons={"Ask": True})

    def failing_ask_question(question: str, *, top_k: int = 5, namespace=None):
        raise RuntimeError("boom")

    app_module.main(fake_streamlit, ask_fn=failing_ask_question)

    assert any(call[0] == "error" and "Unable to answer the question" in call[1] for call in fake_streamlit.calls)


def test_main_renders_monitor_tab_and_calls_orchestrator() -> None:
    fake_monitor_result = app_module.MonitorResult(
        topic="pharmacovigilance",
        selected_sources=["ClinicalTrials.gov", "EMA"],
        items=[
            monitor_module.MonitorItem(
                source="EMA",
                title="EMA update",
                published_date=None,
                category="EMA News",
                description="EMA summary",
                url="https://example.com/ema",
                source_id="ema-1",
            ),
            monitor_module.MonitorItem(
                source="ClinicalTrials.gov",
                title="CT update",
                published_date=None,
                category="INTERVENTIONAL | RECRUITING",
                description="CT summary",
                url="https://example.com/ct",
                source_id="ct-1",
            ),
            ],
            source_errors=[
                monitor_module.MonitorSourceError(source="openFDA", error="timeout", detail="openFDA unavailable"),
            ],
        )
    fake_streamlit = FakeStreamlit(
        session_state={},
        text_inputs={
            "Topic or keyword": "pharmacovigilance",
            "Published after (YYYY-MM-DD, optional)": "2026-08-01",
            "Per-source limit": "3",
            "Keyword filter (optional)": "",
        },
        checkboxes={
            "ClinicalTrials.gov": True,
            "openFDA": False,
            "EMA": True,
            "Show ClinicalTrials.gov": True,
            "Show openFDA": False,
            "Show EMA": True,
        },
        buttons={"Fetch Updates": True},
    )

    with patch.object(app_module, "fetch_monitor_updates", return_value=fake_monitor_result) as fetch_updates:
        app_module.main(fake_streamlit, ask_fn=lambda question: SimpleNamespace(answer="", citations=[]))

    assert fetch_updates.call_count == 1
    assert fetch_updates.call_args.args[0] == "pharmacovigilance"
    assert fetch_updates.call_args.kwargs["selected_sources"] == ["ClinicalTrials.gov", "EMA"]
    assert fetch_updates.call_args.kwargs["per_source_limit"] == 3
    assert fetch_updates.call_args.kwargs["final_limit"] == 3
    assert fake_streamlit.session_state["monitor_result"] is fake_monitor_result
    assert any(call[0] == "subheader" and call[1] == "Monitor Summary" for call in fake_streamlit.calls)
    assert any(call[0] == "warning" and "openFDA: timeout" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "subheader" and call[1] == "Signal Feed" for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "**EMA update**" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "monitor-source-badge--ema" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "[Official source](https://example.com/ema)" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "**CT update**" in call[1] for call in fake_streamlit.calls)


def test_main_uses_session_state_monitor_result_without_refetching() -> None:
    fake_monitor_result = app_module.MonitorResult(
        topic="topic",
        selected_sources=["EMA"],
        items=[
            monitor_module.MonitorItem(
                source="EMA",
                title="EMA item",
                published_date=None,
                category="EMA News",
                description="EMA summary",
                url="https://example.com/ema",
                source_id="ema-1",
            )
        ],
        source_errors=[],
    )
    fake_streamlit = FakeStreamlit(
        session_state={
            "monitor_result": fake_monitor_result,
            "monitor_source_filter": {"ClinicalTrials.gov": False, "openFDA": False, "EMA": True},
            "monitor_keyword_filter": "",
        },
        text_inputs={
            "Topic or keyword": "topic",
            "Published after (YYYY-MM-DD, optional)": "",
            "Per-source limit": "3",
            "Keyword filter (optional)": "nonmatching",
        },
        checkboxes={
            "ClinicalTrials.gov": True,
            "openFDA": False,
            "EMA": True,
            "Show ClinicalTrials.gov": False,
            "Show openFDA": False,
            "Show EMA": True,
        },
        buttons={"Fetch Updates": False},
    )

    with patch.object(app_module, "fetch_monitor_updates") as fetch_updates:
        app_module.main(fake_streamlit, ask_fn=lambda question: SimpleNamespace(answer="", citations=[]))

    assert fetch_updates.call_count == 0
    assert any(call[0] == "subheader" and call[1] == "Monitor Summary" for call in fake_streamlit.calls)
    assert any(call[0] == "info" and "No signals matched the current local filters." in call[1] for call in fake_streamlit.calls)


def test_monitor_item_card_renders_badge_and_collapsible_description() -> None:
    long_description = " ".join(["Long description"] * 30)
    item = monitor_module.MonitorItem(
        source="ClinicalTrials.gov",
        title="Study update",
        published_date=None,
        category="INTERVENTIONAL | RECRUITING",
        description=long_description,
        url="https://example.com/study",
        source_id="NCT12345678",
    )
    fake_streamlit = FakeStreamlit(session_state={})

    app_module._render_monitor_item_card(fake_streamlit, item)

    state_key = app_module._monitor_description_state_key(item)
    assert fake_streamlit.session_state[state_key] is True
    assert any(call[0] == "markdown" and "monitor-source-badge--clinicaltrials" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "write" and long_description in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "button" and call[1] == ("Collapse", f"{state_key}:collapse") for call in fake_streamlit.calls)


def test_monitor_item_card_collapsed_preview_keeps_short_descriptions_visible() -> None:
    short_description = "Short summary of the signal."
    long_description = " ".join(["Long description"] * 30)
    long_item = monitor_module.MonitorItem(
        source="EMA",
        title="EMA item",
        published_date=None,
        category="EMA News",
        description=long_description,
        url="https://example.com/ema",
        source_id="ema-1",
    )
    short_item = monitor_module.MonitorItem(
        source="openFDA",
        title="openFDA item",
        published_date=None,
        category="Drug Adverse Event",
        description=short_description,
        url="https://example.com/openfda",
        source_id="fda-1",
    )

    long_state_key = app_module._monitor_description_state_key(long_item)
    fake_streamlit = FakeStreamlit(
        session_state={long_state_key: False},
        buttons={f"{long_state_key}:expand": True},
    )

    app_module._render_monitor_item_card(fake_streamlit, long_item)
    app_module._render_monitor_item_card(fake_streamlit, short_item)

    assert any(call[0] == "write" and "Long description Long description" in call[1] and "…" in call[1] for call in fake_streamlit.calls)
    assert any(call[0] == "write" and short_description == call[1] for call in fake_streamlit.calls)
    assert not any(call[0] == "button" and call[1][0] == "Collapse" and call[1][1] == f"{long_state_key}:collapse" for call in fake_streamlit.calls)

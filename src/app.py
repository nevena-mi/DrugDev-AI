"""Streamlit application for the Ask/ Learn/ Monitor interface."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from src.chatbot import ask_question


logger = logging.getLogger(__name__)


class StreamlitLike(Protocol):
    """Minimal Streamlit surface used by the application."""

    def set_page_config(self, **kwargs: Any) -> None: ...
    def title(self, text: str) -> None: ...
    def markdown(self, text: str) -> None: ...
    def write(self, value: Any) -> None: ...
    def subheader(self, text: str) -> None: ...
    def text_input(self, label: str, value: str = "", placeholder: str = "") -> str: ...
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

    for index, citation in enumerate(citations, start=1):
        citation_data = _to_mapping(citation)
        title = str(citation_data.get("document_title", "")).strip()
        if not title or title in seen_titles:
            continue

        seen_titles.add(title)
        rendered_titles.append(title)
        st.markdown(f"**Citation {len(rendered_titles)}**: {title}")


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
        _render_placeholder(ui, "Learn")

    with monitor_tab:
        _render_placeholder(ui, "Monitor")


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()

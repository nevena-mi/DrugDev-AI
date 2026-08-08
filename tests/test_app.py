from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import src.app as app_module


@dataclass
class FakeTab:
    label: str

    def __enter__(self) -> "FakeTab":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeStreamlit:
    def __init__(self, *, question: str, button_pressed: bool = True) -> None:
        self.question = question
        self.button_pressed = button_pressed
        self.calls: list[tuple[str, object]] = []

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, text: str) -> None:
        self.calls.append(("title", text))

    def markdown(self, text: str) -> None:
        self.calls.append(("markdown", text))

    def write(self, value) -> None:
        self.calls.append(("write", value))

    def subheader(self, text: str) -> None:
        self.calls.append(("subheader", text))

    def text_input(self, label: str, value: str = "", placeholder: str = "") -> str:
        self.calls.append(("text_input", (label, value, placeholder)))
        return self.question

    def button(self, label: str) -> bool:
        self.calls.append(("button", label))
        return self.button_pressed

    def error(self, text: str) -> None:
        self.calls.append(("error", text))

    def success(self, text: str) -> None:
        self.calls.append(("success", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def divider(self) -> None:
        self.calls.append(("divider", None))

    def tabs(self, labels: list[str]) -> list[FakeTab]:
        self.calls.append(("tabs", tuple(labels)))
        return [FakeTab(label) for label in labels]


def test_app_imports_cleanly() -> None:
    assert hasattr(app_module, "main")


def test_main_renders_ask_and_placeholders() -> None:
    ask_calls: list[tuple[str, dict[str, object]]] = []
    fake_result = SimpleNamespace(
        answer="Grounded answer",
        citations=[
            SimpleNamespace(
                id="ema/example.pdf::chunk-0",
                filename="example.pdf",
                relative_file_path="ema/example.pdf",
                source_organization="ema",
                document_title="example",
                chunk_id="ema/example.pdf::chunk-0",
                score=0.91,
            )
        ],
        retrieved_chunks=[
            SimpleNamespace(
                id="ema/example.pdf::chunk-0",
                score=0.91,
                metadata={
                    "filename": "example.pdf",
                    "relative_file_path": "ema/example.pdf",
                    "source_organization": "ema",
                    "document_title": "example",
                    "chunk_id": "ema/example.pdf::chunk-0",
                },
            )
        ],
    )

    fake_streamlit = FakeStreamlit(question="What is GCP?")

    def fake_ask_question(question: str, *, top_k: int = 5, namespace=None):
        ask_calls.append(("ask", {"question": question, "top_k": top_k, "namespace": namespace}))
        assert question == "What is GCP?"
        assert top_k == 5
        assert namespace is None
        return fake_result

    app_module.main(fake_streamlit, ask_fn=fake_ask_question)

    assert ("tabs", ("Ask", "Learn", "Monitor")) in fake_streamlit.calls
    assert len(ask_calls) == 1
    assert ("subheader", "Answer") in fake_streamlit.calls
    assert ("subheader", "Citations") in fake_streamlit.calls
    assert ("subheader", "Retrieved Sources") in fake_streamlit.calls
    assert ("write", "Grounded answer") in fake_streamlit.calls
    assert any(call[0] == "markdown" and "Citation 1" in call[1] for call in fake_streamlit.calls)
    assert sum(1 for call in fake_streamlit.calls if call[0] == "info" and "later phase" in call[1].lower()) == 2


def test_main_handles_backend_errors_gracefully() -> None:
    fake_streamlit = FakeStreamlit(question="What is GCP?")

    def failing_ask_question(question: str, *, top_k: int = 5, namespace=None):
        raise RuntimeError("boom")

    app_module.main(fake_streamlit, ask_fn=failing_ask_question)

    assert any(call[0] == "error" and "Unable to answer the question" in call[1] for call in fake_streamlit.calls)

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

import src.monitor as monitor_module


def _make_item(
    *,
    source: str,
    source_id: str,
    title: str,
    published_date: date | None,
) -> monitor_module.MonitorItem:
    return monitor_module.MonitorItem(
        source=source,
        title=title,
        published_date=published_date,
        category="Test",
        description=f"{title} description",
        url=f"https://example.com/{source_id}",
        source_id=source_id,
    )


class FakeSourceError(RuntimeError):
    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass
class FakeAdapter:
    source_name: str
    items: list[monitor_module.MonitorItem]
    calls: list[tuple[str, date | None, date | None, int]]
    error: Exception | None = None

    def fetch(
        self,
        topic: str,
        *,
        published_after: date | None = None,
        published_before: date | None = None,
        limit: int = 20,
    ) -> list[monitor_module.MonitorItem]:
        self.calls.append((topic, published_after, published_before, limit))
        if self.error is not None:
            raise self.error
        return list(self.items)


def _make_registry(adapters: dict[str, FakeAdapter]) -> dict[str, object]:
    def _factory(adapter: FakeAdapter):
        def _registration():
            return adapter, (FakeSourceError,)

        return _registration

    registry: dict[str, object] = {}
    for name, adapter in adapters.items():
        registry[name] = _factory(adapter)
    return registry


def test_orchestrator_calls_selected_sources_and_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = FakeAdapter(
        source_name="ClinicalTrials.gov",
        items=[_make_item(source="ClinicalTrials.gov", source_id="ct-1", title="CT", published_date=date(2026, 8, 10))],
        calls=[],
    )
    openfda = FakeAdapter(
        source_name="openFDA",
        items=[_make_item(source="openFDA", source_id="fd-1", title="FDA", published_date=date(2026, 8, 9))],
        calls=[],
    )
    ema = FakeAdapter(
        source_name="EMA",
        items=[_make_item(source="EMA", source_id="ema-1", title="EMA", published_date=date(2026, 8, 8))],
        calls=[],
    )

    monkeypatch.setattr(
        monitor_module,
        "_SOURCE_REGISTRY",
        _make_registry(
            {
                "ClinicalTrials.gov": clinical,
                "openFDA": openfda,
                "EMA": ema,
            }
        ),
    )

    result = monitor_module.fetch_monitor_updates(
        " Parkinson ",
        selected_sources=["ClinicalTrials.gov", "EMA", "ClinicalTrials.gov"],
        published_after=date(2026, 8, 1),
        published_before=date(2026, 8, 31),
        per_source_limit=7,
    )

    assert clinical.calls == [("Parkinson", date(2026, 8, 1), date(2026, 8, 31), 7)]
    assert openfda.calls == []
    assert ema.calls == [("Parkinson", date(2026, 8, 1), date(2026, 8, 31), 7)]
    assert result.selected_sources == ["ClinicalTrials.gov", "EMA"]
    assert result.topic == "Parkinson"


def test_orchestrator_combines_deduplicates_sorts_and_applies_final_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    shared_new = _make_item(source="EMA", source_id="shared-new", title="Shared new", published_date=date(2026, 8, 12))
    shared_old = _make_item(source="EMA", source_id="shared-old", title="Shared old", published_date=date(2026, 8, 1))
    clinical = FakeAdapter(
        source_name="ClinicalTrials.gov",
        items=[
            shared_old,
            _make_item(source="ClinicalTrials.gov", source_id="ct-2", title="CT newer", published_date=date(2026, 8, 15)),
        ],
        calls=[],
    )
    openfda = FakeAdapter(
        source_name="openFDA",
        items=[
            shared_old,
            shared_new,
            _make_item(source="openFDA", source_id="fd-1", title="FDA middle", published_date=date(2026, 8, 10)),
        ],
        calls=[],
    )

    monkeypatch.setattr(
        monitor_module,
        "_SOURCE_REGISTRY",
        _make_registry(
            {
                "ClinicalTrials.gov": clinical,
                "openFDA": openfda,
            }
        ),
    )

    result = monitor_module.fetch_monitor_updates(
        "topic",
        selected_sources=["openFDA", "ClinicalTrials.gov"],
        per_source_limit=10,
        final_limit=2,
    )

    assert [item.source_id for item in result.items] == ["ct-2", "shared-new"]
    assert [item.published_date for item in result.items] == [date(2026, 8, 15), date(2026, 8, 12)]


def test_orchestrator_records_partial_failures_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = FakeAdapter(
        source_name="ClinicalTrials.gov",
        items=[],
        calls=[],
        error=FakeSourceError("boom", detail="clinical trials unavailable"),
    )
    working = FakeAdapter(
        source_name="EMA",
        items=[_make_item(source="EMA", source_id="ema-1", title="EMA item", published_date=date(2026, 8, 11))],
        calls=[],
    )

    monkeypatch.setattr(
        monitor_module,
        "_SOURCE_REGISTRY",
        _make_registry(
            {
                "ClinicalTrials.gov": failing,
                "EMA": working,
            }
        ),
    )

    result = monitor_module.fetch_monitor_updates(
        "topic",
        selected_sources=["ClinicalTrials.gov", "EMA"],
        per_source_limit=5,
    )

    assert result.items and result.items[0].source == "EMA"
    assert len(result.source_errors) == 1
    assert result.source_errors[0].source == "ClinicalTrials.gov"
    assert result.source_errors[0].error == "boom"
    assert "clinical trials unavailable" in (result.source_errors[0].detail or "")


def test_orchestrator_returns_valid_result_when_all_sources_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_ct = FakeAdapter(
        source_name="ClinicalTrials.gov",
        items=[],
        calls=[],
        error=FakeSourceError("ct failed", detail="ct detail"),
    )
    failing_ema = FakeAdapter(
        source_name="EMA",
        items=[],
        calls=[],
        error=FakeSourceError("ema failed", detail="ema detail"),
    )

    monkeypatch.setattr(
        monitor_module,
        "_SOURCE_REGISTRY",
        _make_registry(
            {
                "ClinicalTrials.gov": failing_ct,
                "EMA": failing_ema,
            }
        ),
    )

    result = monitor_module.fetch_monitor_updates(
        "topic",
        selected_sources=["ClinicalTrials.gov", "EMA"],
        per_source_limit=5,
    )

    assert result.items == []
    assert [error.source for error in result.source_errors] == ["ClinicalTrials.gov", "EMA"]


def test_orchestrator_rejects_invalid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor_module, "_SOURCE_REGISTRY", {})

    with pytest.raises(monitor_module.MonitorValidationError, match="topic must be a non-empty string"):
        monitor_module.fetch_monitor_updates("", selected_sources=["EMA"])

    with pytest.raises(monitor_module.MonitorValidationError, match="at least one supported source"):
        monitor_module.fetch_monitor_updates("topic", selected_sources=[])

    with pytest.raises(monitor_module.MonitorValidationError, match="unsupported Monitor source"):
        monitor_module.fetch_monitor_updates("topic", selected_sources=["EMA"])

    monkeypatch.setattr(
        monitor_module,
        "_SOURCE_REGISTRY",
        {"EMA": lambda: (FakeAdapter("EMA", [], []), (FakeSourceError,))},
    )

    with pytest.raises(monitor_module.MonitorValidationError, match="per_source_limit"):
        monitor_module.fetch_monitor_updates("topic", selected_sources=["EMA"], per_source_limit=0)

    with pytest.raises(monitor_module.MonitorValidationError, match="final_limit"):
        monitor_module.fetch_monitor_updates("topic", selected_sources=["EMA"], final_limit=0)

    with pytest.raises(monitor_module.MonitorValidationError, match="published_after cannot be later"):
        monitor_module.fetch_monitor_updates(
            "topic",
            selected_sources=["EMA"],
            published_after=date(2026, 8, 2),
            published_before=date(2026, 8, 1),
        )

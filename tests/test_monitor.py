from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import src.monitor as monitor_module


def _make_item(
    *,
    source: str = "EMA",
    title: str = "EMA scientific guideline update",
    published_date: date | None = date(2026, 8, 10),
    category: str = "Guidance",
    description: str = "Updated scientific advice for regulatory review.",
    url: str = "https://example.com/ema-guidance",
    source_id: str = "ema-001",
) -> monitor_module.MonitorItem:
    return monitor_module.MonitorItem(
        source=source,
        title=title,
        published_date=published_date,
        category=category,
        description=description,
        url=url,
        source_id=source_id,
    )


def test_monitor_item_construction_and_result_container() -> None:
    item = _make_item()
    result = monitor_module.MonitorResult(
        topic="pharmacovigilance",
        selected_sources=["EMA", "FDA"],
        items=[item],
        source_errors=[monitor_module.MonitorSourceError(source="FDA", error="timeout")],
    )

    assert item.source == "EMA"
    assert item.title == "EMA scientific guideline update"
    assert item.published_date == date(2026, 8, 10)
    assert result.topic == "pharmacovigilance"
    assert result.query == "pharmacovigilance"
    assert result.selected_sources == ["EMA", "FDA"]
    assert result.items == [item]
    assert result.source_errors == [monitor_module.MonitorSourceError(source="FDA", error="timeout")]


def test_sort_monitor_items_newest_first_and_missing_dates_last() -> None:
    items = [
        _make_item(source_id="old", published_date=date(2026, 8, 1)),
        _make_item(source_id="missing", published_date=None),
        _make_item(source_id="new", published_date=date(2026, 8, 15)),
    ]

    sorted_items = monitor_module.sort_monitor_items_newest_first(items)

    assert [item.source_id for item in sorted_items] == ["new", "old", "missing"]


def test_filter_monitor_items_is_case_insensitive() -> None:
    items = [
        _make_item(title="EMA updates on Pharmacovigilance", description="Safety signal management"),
        _make_item(title="FDA device communication", description="Unrelated content"),
    ]

    filtered = monitor_module.filter_monitor_items(items, "pharmacovigilance")

    assert [item.source_id for item in filtered] == ["ema-001"]


def test_filter_monitor_items_can_match_multiple_keywords() -> None:
    items = [
        _make_item(title="EMA pharmacovigilance update", description="Safety signal management"),
        _make_item(title="EMA pharmacovigilance update", description="Different category", source_id="ema-002"),
        _make_item(title="EMA device update", description="Different content", source_id="ema-003"),
    ]

    filtered = monitor_module.filter_monitor_items(items, ["EMA", "safety"])

    assert [item.source_id for item in filtered] == ["ema-001"]


def test_deduplicate_monitor_items_by_source_and_source_id_preserves_first_occurrence() -> None:
    first = _make_item(source_id="dup-1", description="First version")
    duplicate = _make_item(source_id="dup-1", description="Duplicate version")
    second = _make_item(source="FDA", source_id="dup-1", description="Same id but different source")

    deduplicated = monitor_module.deduplicate_monitor_items([first, duplicate, second, first])

    assert deduplicated == [first, second]


def test_source_errors_can_coexist_with_successful_items() -> None:
    item = _make_item()
    result = monitor_module.MonitorResult(
        topic="regulatory updates",
        selected_sources=["EMA", "FDA"],
        items=[item],
        source_errors=[
            monitor_module.MonitorSourceError(
                source="FDA",
                error="timeout",
                detail="ClinicalTrials.gov adapter timed out",
            )
        ],
    )

    assert result.items == [item]
    assert result.source_errors[0].source == "FDA"
    assert result.source_errors[0].error == "timeout"
    assert "timed out" in (result.source_errors[0].detail or "")


def test_dummy_source_adapter_conforms_to_shared_interface() -> None:
    @dataclass
    class DummyAdapter:
        def fetch(
            self,
            topic: str,
            *,
            published_after: date | None = None,
            published_before: date | None = None,
            limit: int = 20,
        ) -> list[monitor_module.MonitorItem]:
            return [_make_item(title=topic, source_id=str(limit))]

    adapter = DummyAdapter()

    assert isinstance(adapter, monitor_module.MonitorSourceAdapter)
    assert adapter.fetch("monitoring") == [_make_item(title="monitoring", source_id="20")]

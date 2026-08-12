"""Shared Monitor-mode data model and utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, Sequence, runtime_checkable


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MonitorItem:
    """A normalized regulatory signal returned by a Monitor source."""

    source: str
    title: str
    published_date: date | None
    category: str
    description: str
    url: str
    source_id: str


@dataclass(slots=True)
class MonitorSourceError:
    """Structured error information for a single Monitor source."""

    source: str
    error: str
    detail: str | None = None


@dataclass(slots=True)
class MonitorResult:
    """Container for a Monitor request and its normalized source output."""

    topic: str
    selected_sources: list[str] = field(default_factory=list)
    items: list[MonitorItem] = field(default_factory=list)
    source_errors: list[MonitorSourceError] = field(default_factory=list)

    @property
    def query(self) -> str:
        """Backward-compatible alias for the monitored topic."""

        return self.topic


@runtime_checkable
class MonitorSourceAdapter(Protocol):
    """Contract for future Monitor source adapters."""

    def fetch(
        self,
        topic: str,
        *,
        published_after: date | None = None,
        published_before: date | None = None,
        limit: int = 20,
    ) -> list[MonitorItem]:
        """Fetch normalized Monitor items for a topic."""


def sort_monitor_items_newest_first(items: Sequence[MonitorItem]) -> list[MonitorItem]:
    """Return items sorted from newest to oldest, keeping missing dates last."""

    return sorted(items, key=lambda item: item.published_date or date.min, reverse=True)


def _normalize_keywords(keywords: Sequence[str] | str | None) -> list[str]:
    if keywords is None:
        return []
    if isinstance(keywords, str):
        candidate_values = [keywords]
    else:
        candidate_values = list(keywords)
    return [value.strip().lower() for value in candidate_values if isinstance(value, str) and value.strip()]


def filter_monitor_items(
    items: Sequence[MonitorItem],
    keywords: Sequence[str] | str | None,
) -> list[MonitorItem]:
    """Filter items by case-insensitive keyword presence across text fields."""

    normalized_keywords = _normalize_keywords(keywords)
    if not normalized_keywords:
        return list(items)

    filtered_items: list[MonitorItem] = []
    for item in items:
        searchable_text = " ".join(
            [
                item.source,
                item.title,
                item.category,
                item.description,
                item.url,
                item.source_id,
            ]
        ).lower()
        if all(keyword in searchable_text for keyword in normalized_keywords):
            filtered_items.append(item)
    return filtered_items


def deduplicate_monitor_items(items: Sequence[MonitorItem]) -> list[MonitorItem]:
    """Deduplicate items by source and source_id while preserving order."""

    deduplicated_items: list[MonitorItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.source, item.source_id)
        if key in seen:
            continue
        seen.add(key)
        deduplicated_items.append(item)
    return deduplicated_items

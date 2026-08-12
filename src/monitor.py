"""Shared Monitor-mode data model and utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Protocol, Sequence, runtime_checkable


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


class MonitorValidationError(ValueError):
    """Raised when a Monitor orchestration request is invalid."""


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


SourceRegistration = Callable[[], tuple[MonitorSourceAdapter, tuple[type[Exception], ...]]]


def _get_clinical_trials_registration() -> tuple[MonitorSourceAdapter, tuple[type[Exception], ...]]:
    from src.monitor_sources.clinical_trials import ClinicalTrialsAdapterError, ClinicalTrialsGovAdapter

    return ClinicalTrialsGovAdapter(), (ClinicalTrialsAdapterError,)


def _get_openfda_registration() -> tuple[MonitorSourceAdapter, tuple[type[Exception], ...]]:
    from src.monitor_sources.openfda import OpenFDAAdapter, OpenFDAAdapterError

    return OpenFDAAdapter(), (OpenFDAAdapterError,)


def _get_ema_registration() -> tuple[MonitorSourceAdapter, tuple[type[Exception], ...]]:
    from src.monitor_sources.ema import EMAAdapter, EMAAdapterError

    return EMAAdapter(), (EMAAdapterError,)


_SOURCE_REGISTRY: dict[str, SourceRegistration] = {
    "ClinicalTrials.gov": _get_clinical_trials_registration,
    "openFDA": _get_openfda_registration,
    "EMA": _get_ema_registration,
}


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


def _normalize_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise MonitorValidationError("topic must be a string")
    normalized_topic = topic.strip()
    if not normalized_topic:
        raise MonitorValidationError("topic must be a non-empty string")
    return normalized_topic


def _normalize_selected_sources(selected_sources: Sequence[str]) -> list[str]:
    if not isinstance(selected_sources, Sequence) or isinstance(selected_sources, str):
        raise MonitorValidationError("selected_sources must be a sequence of source names")

    normalized_sources: list[str] = []
    seen: set[str] = set()
    for source_name in selected_sources:
        if not isinstance(source_name, str):
            raise MonitorValidationError("selected_sources must contain only strings")
        canonical_source = source_name.strip()
        if not canonical_source:
            raise MonitorValidationError("selected_sources cannot contain empty source names")
        if canonical_source in seen:
            continue
        if canonical_source not in _SOURCE_REGISTRY:
            raise MonitorValidationError(f"unsupported Monitor source: {canonical_source}")
        seen.add(canonical_source)
        normalized_sources.append(canonical_source)

    if not normalized_sources:
        raise MonitorValidationError("at least one supported source must be selected")
    return normalized_sources


def _validate_limits(
    *,
    per_source_limit: int,
    final_limit: int | None,
) -> None:
    if per_source_limit <= 0:
        raise MonitorValidationError("per_source_limit must be a positive integer")
    if final_limit is not None and final_limit <= 0:
        raise MonitorValidationError("final_limit must be a positive integer when provided")


def _validate_date_bounds(
    *,
    published_after: date | None,
    published_before: date | None,
) -> None:
    if published_after is not None and published_before is not None and published_after > published_before:
        raise MonitorValidationError("published_after cannot be later than published_before")


def fetch_monitor_updates(
    topic: str,
    *,
    selected_sources: Sequence[str],
    published_after: date | None = None,
    published_before: date | None = None,
    per_source_limit: int = 20,
    final_limit: int | None = None,
) -> MonitorResult:
    """Fetch, merge, deduplicate, and sort Monitor updates from selected sources."""

    normalized_topic = _normalize_topic(topic)
    normalized_sources = _normalize_selected_sources(selected_sources)
    _validate_limits(per_source_limit=per_source_limit, final_limit=final_limit)
    _validate_date_bounds(published_after=published_after, published_before=published_before)

    logger.info(
        "Fetching Monitor updates for topic=%r from sources=%s",
        normalized_topic,
        normalized_sources,
    )

    collected_items: list[MonitorItem] = []
    source_errors: list[MonitorSourceError] = []

    for source_name in normalized_sources:
        registration = _SOURCE_REGISTRY[source_name]
        adapter, error_types = registration()
        logger.debug(
            "Querying Monitor source %s with per_source_limit=%s after=%s before=%s",
            source_name,
            per_source_limit,
            published_after,
            published_before,
        )
        try:
            source_items = adapter.fetch(
                normalized_topic,
                published_after=published_after,
                published_before=published_before,
                limit=per_source_limit,
            )
        except error_types as exc:
            source_errors.append(
                MonitorSourceError(
                    source=source_name,
                    error=str(exc),
                    detail=getattr(exc, "detail", None),
                )
            )
            logger.warning(
                "Monitor source %s failed: %s",
                source_name,
                exc,
            )
            continue

        collected_items.extend(source_items)

    normalized_items = deduplicate_monitor_items(collected_items)
    normalized_items = sort_monitor_items_newest_first(normalized_items)
    if final_limit is not None:
        normalized_items = normalized_items[:final_limit]

    result = MonitorResult(
        topic=normalized_topic,
        selected_sources=normalized_sources,
        items=normalized_items,
        source_errors=source_errors,
    )
    logger.info(
        "Monitor updates completed for topic=%r with %s item(s) and %s source error(s)",
        normalized_topic,
        len(result.items),
        len(result.source_errors),
    )
    return result

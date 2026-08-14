"""Token-usage and estimated-cost analytics for DrugDev-AI.

This module is independent from Streamlit and application workflows.
API call sites can record usage here, while the UI can read summaries later.

Important:
- Token counts come from provider response metadata where available.
- Costs are estimates based on the pricing table below.
- Unknown models/providers are still logged; their cost remains None.
- API keys and prompt/response content are never stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_COST_LOG_PATH = Path(
    os.getenv("DRUGDEV_COST_LOG_PATH", "data/cost_events.jsonl")
)

CostPhase = Literal["build", "runtime", "development"]


@dataclass(frozen=True)
class OpenAIPrice:
    """OpenAI token prices in USD per 1 million tokens."""

    input_per_million: float
    output_per_million: float = 0.0
    cached_input_per_million: float | None = None


# Update this table when provider pricing changes.
# Checked against official OpenAI pricing/docs on 2026-08-14.
OPENAI_PRICING: dict[str, OpenAIPrice] = {
    "gpt-4o-mini": OpenAIPrice(0.15, 0.60, 0.075),
    "gpt-5.6-sol": OpenAIPrice(5.00, 30.00, 0.50),
    "gpt-5.6-terra": OpenAIPrice(2.50, 15.00, 0.25),
    "gpt-5.6-luna": OpenAIPrice(1.00, 6.00, 0.10),
    "text-embedding-3-small": OpenAIPrice(0.02),
    "text-embedding-3-large": OpenAIPrice(0.13),
    "text-embedding-ada-002": OpenAIPrice(0.10),
}


@dataclass
class CostEvent:
    """One billable or measurable AI operation."""

    timestamp: str
    phase: CostPhase
    mode: str
    operation: str
    provider: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    search_units: float = 0.0
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class CostSummary:
    """Aggregated usage suitable for a dashboard."""

    event_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    search_units: float
    estimated_cost_usd: float
    events_without_known_price: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_value(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _as_non_negative_float(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(parsed, 0.0)


def estimate_openai_cost(
    model: str,
    *,
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> float | None:
    """Estimate OpenAI request cost from token counts."""

    price = OPENAI_PRICING.get(model)
    if price is None:
        return None

    input_tokens = max(input_tokens, 0)
    cached_input_tokens = min(max(cached_input_tokens, 0), input_tokens)
    output_tokens = max(output_tokens, 0)
    uncached_input = input_tokens - cached_input_tokens

    input_cost = uncached_input / 1_000_000 * price.input_per_million
    output_cost = output_tokens / 1_000_000 * price.output_per_million
    cached_rate = (
        price.cached_input_per_million
        if price.cached_input_per_million is not None
        else price.input_per_million
    )
    cached_cost = cached_input_tokens / 1_000_000 * cached_rate
    return input_cost + cached_cost + output_cost


def append_cost_event(
    event: CostEvent,
    *,
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> None:
    """Append an event to the local JSONL usage ledger."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def record_openai_response(
    response: Any,
    *,
    phase: CostPhase = "runtime",
    mode: str,
    operation: str,
    model: str | None = None,
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> CostEvent | None:
    """Record token usage from an OpenAI Responses API response."""

    usage = _get_value(response, "usage", {}) or {}
    input_tokens = _as_non_negative_int(_get_value(usage, "input_tokens", 0))
    output_tokens = _as_non_negative_int(_get_value(usage, "output_tokens", 0))
    input_details = _get_value(usage, "input_tokens_details", {}) or {}
    cached_input_tokens = _as_non_negative_int(
        _get_value(input_details, "cached_tokens", 0)
    )

    # Ignore mocked/test responses without real usage.
    if not usage or (
        input_tokens == 0
        and cached_input_tokens == 0
        and output_tokens == 0
    ):
        return None 


    resolved_model = str(
        model or _get_value(response, "model", "") or "unknown"
    ).strip()

    event = CostEvent(
        timestamp=_utc_timestamp(),
        phase=phase,
        mode=mode,
        operation=operation,
        provider="OpenAI",
        model=resolved_model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_openai_cost(
            resolved_model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        ),
    )
    append_cost_event(event, path=path)
    return event


def record_openai_embedding(
    response: Any,
    *,
    phase: CostPhase,
    mode: str,
    model: str,
    operation: str = "embedding",
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> CostEvent | None:
    """Record usage from an OpenAI embeddings response."""

    usage = _get_value(response, "usage", {}) or {}
    if not usage:
        return None

    input_tokens = _as_non_negative_int(
        _get_value(
            usage,
            "prompt_tokens",
            _get_value(usage, "input_tokens", _get_value(usage, "total_tokens", 0)),
        )
    )
    event = CostEvent(
        timestamp=_utc_timestamp(),
        mode=mode,
        phase=phase,
        operation=operation,
        provider="OpenAI",
        model=model,
        input_tokens=input_tokens,
        estimated_cost_usd=estimate_openai_cost(model, input_tokens=input_tokens),
    )
    append_cost_event(event, path=path)
    return event


def record_cohere_rerank(
    response: Any,
    *,
    phase: CostPhase = "runtime",
    mode: str,
    model: str,
    operation: str = "rerank",
    usd_per_1000_search_units: float | None = None,
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> CostEvent:
    """Record Cohere rerank usage.

    Rerank usage is measured in search units. Supply your account's current
    USD price per 1,000 search units to calculate money cost; otherwise the
    usage is logged with unknown monetary cost.
    """

    meta = _get_value(response, "meta", {}) or {}
    billed_units = _get_value(meta, "billed_units", {}) or {}
    search_units = _as_non_negative_float(
        _get_value(billed_units, "search_units", 0)
    )

    estimated_cost = None
    if usd_per_1000_search_units is not None:
        estimated_cost = search_units / 1_000 * max(
            float(usd_per_1000_search_units), 0.0
        )

    event = CostEvent(
        timestamp=_utc_timestamp(),
        mode=mode,
        phase=phase,
        operation=operation,
        provider="Cohere",
        model=model,
        search_units=search_units,
        estimated_cost_usd=estimated_cost,
    )
    append_cost_event(event, path=path)
    return event


def load_cost_events(
    *,
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> list[CostEvent]:
    """Load all valid events from the local cost ledger."""

    source = Path(path)
    if not source.exists():
        return []

    events: list[CostEvent] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(CostEvent(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return events


def summarize_cost_events(
    events: Iterable[CostEvent],
    *,
    mode: str | None = None,
    provider: str | None = None,
) -> CostSummary:
    """Aggregate usage, optionally filtered by mode/provider."""

    selected = [
        event
        for event in events
        if (mode is None or event.mode == mode)
        and (provider is None or event.provider == provider)
    ]

    input_tokens = sum(event.input_tokens for event in selected)
    cached_input_tokens = sum(event.cached_input_tokens for event in selected)
    output_tokens = sum(event.output_tokens for event in selected)
    search_units = sum(event.search_units for event in selected)
    known_cost = sum(event.estimated_cost_usd or 0.0 for event in selected)
    unknown_price_count = sum(
        1 for event in selected if event.estimated_cost_usd is None
    )

    return CostSummary(
        event_count=len(selected),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        search_units=search_units,
        estimated_cost_usd=known_cost,
        events_without_known_price=unknown_price_count,
    )


def summarize_by_mode(events: Iterable[CostEvent]) -> dict[str, CostSummary]:
    """Return separate totals for Ask, Learn, Monitor, etc."""

    materialized = list(events)
    return {
        mode: summarize_cost_events(materialized, mode=mode)
        for mode in sorted({event.mode for event in materialized})
    }


def summarize_by_operation(events: Iterable[CostEvent]) -> dict[str, CostSummary]:
    """Return totals grouped by operation."""

    materialized = list(events)
    operations = sorted({event.operation for event in materialized})
    return {
        operation: summarize_cost_events(
            [event for event in materialized if event.operation == operation]
        )
        for operation in operations
    }


def clear_cost_events(
    *,
    path: str | Path = DEFAULT_COST_LOG_PATH,
) -> None:
    """Delete the local cost ledger, if present."""

    target = Path(path)
    if target.exists():
        target.unlink()

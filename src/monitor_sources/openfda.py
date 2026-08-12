"""openFDA Drug Adverse Event Monitor source adapter."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

from src.config import OPENFDA_API_KEY
from src.monitor import MonitorItem, MonitorSourceAdapter, sort_monitor_items_newest_first


logger = logging.getLogger(__name__)

OPENFDA_API_BASE_URL = "https://api.fda.gov/drug/event.json"
OPENFDA_ENDPOINT_DOC_URL = "https://open.fda.gov/apis/drug/event/"
HTTP_TIMEOUT_SECONDS = 10.0
MAX_LIMIT = 1000


class OpenFDAAdapterError(RuntimeError):
    """Raised when the openFDA adapter cannot fetch or parse data."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _nested_get(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        flattened: list[str] = []
        for key in ("value", "name", "text", "title", "description", "term"):
            flattened.extend(_flatten_text(value.get(key)))
        if flattened:
            return flattened
        for nested_value in value.values():
            flattened.extend(_flatten_text(nested_value))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened: list[str] = []
        for nested_value in value:
            flattened.extend(_flatten_text(nested_value))
        return flattened
    text = str(value).strip()
    return [text] if text else []


def _first_text(value: Any) -> str | None:
    flattened = _flatten_text(value)
    return flattened[0] if flattened else None


def _unique_texts(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        unique.append(normalized)
    return unique


def _parse_date(value: Any) -> date | None:
    text = _first_text(value)
    if not text:
        return None

    for date_format in ("%Y%m%d", "%Y-%m-%d", "%Y%m", "%Y"):
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        if date_format == "%Y":
            return date(parsed.year, 1, 1)
        if date_format == "%Y%m":
            return date(parsed.year, parsed.month, 1)
        return parsed.date()

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _escape_search_term(topic: str) -> str:
    return topic.strip().replace("\\", "\\\\").replace('"', '\\"')


def _build_search_query(topic: str) -> str:
    escaped_topic = _escape_search_term(topic)
    quoted_topic = f'"{escaped_topic}"'
    clauses = [
        f"patient.drug.medicinalproduct:{quoted_topic}",
        f"patient.drug.openfda.brand_name:{quoted_topic}",
        f"patient.drug.openfda.generic_name:{quoted_topic}",
        f"patient.reaction.reactionmeddrapt:{quoted_topic}",
    ]
    return " OR ".join(clauses)


def _build_request_url(topic: str, *, limit: int) -> str:
    params: list[tuple[str, str]] = [
        ("search", f"({_build_search_query(topic)})"),
        ("sort", "receivedate:desc"),
        ("limit", str(min(max(limit, 1), MAX_LIMIT))),
    ]
    if OPENFDA_API_KEY:
        params.append(("api_key", OPENFDA_API_KEY))
    return f"{OPENFDA_API_BASE_URL}?{parse.urlencode(params, quote_via=parse.quote_plus)}"


def _read_json(url: str) -> dict[str, Any]:
    request_obj = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "DrugDev-AI-Monitor/1.0",
        },
    )
    try:
        with request.urlopen(request_obj, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", response.getcode())
            raw_body = response.read()
    except urlerror.HTTPError as exc:
        detail = None
        if exc.fp is not None:
            try:
                detail = exc.fp.read().decode("utf-8", errors="replace").strip() or None
            except Exception:  # pragma: no cover - defensive
                detail = None
        raise OpenFDAAdapterError(
            f"openFDA returned HTTP {exc.code}",
            detail=detail,
        ) from exc
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise OpenFDAAdapterError(
            "openFDA request failed",
            detail=str(exc),
        ) from exc

    if status != 200:
        raise OpenFDAAdapterError(
            f"openFDA returned HTTP {status}",
            detail=raw_body.decode("utf-8", errors="replace").strip() or None,
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenFDAAdapterError(
            "openFDA returned malformed JSON",
            detail=str(exc),
        ) from exc

    if not isinstance(payload, dict):
        raise OpenFDAAdapterError(
            "openFDA returned an unexpected response payload",
            detail=f"Expected a JSON object, got {type(payload).__name__}",
        )
    return payload


def _extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [result for result in results if isinstance(result, dict)]
    raise OpenFDAAdapterError(
        "openFDA returned an unexpected response structure",
        detail="Missing results list in response payload",
    )


def _passes_date_filter(
    report_date: date | None,
    *,
    published_after: date | None,
    published_before: date | None,
) -> bool:
    if report_date is None:
        return published_after is None and published_before is None
    if published_after is not None and report_date < published_after:
        return False
    if published_before is not None and report_date > published_before:
        return False
    return True


def _extract_drug_names(result: dict[str, Any]) -> list[str]:
    drugs = _nested_get(result, "patient", "drug")
    names: list[str] = []
    if not isinstance(drugs, list):
        return names

    for drug in drugs:
        if not isinstance(drug, dict):
            continue
        candidate_fields = [
            drug.get("medicinalproduct"),
            drug.get("drugname"),
            _nested_get(drug, "openfda", "brand_name"),
            _nested_get(drug, "openfda", "generic_name"),
            _nested_get(drug, "openfda", "substance_name"),
        ]
        for field_value in candidate_fields:
            text = _first_text(field_value)
            if text:
                names.append(text)
                break
    return _unique_texts(names)


def _extract_reaction_terms(result: dict[str, Any]) -> list[str]:
    reactions = _nested_get(result, "patient", "reaction")
    terms: list[str] = []
    if not isinstance(reactions, list):
        return terms

    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        text = _first_text(reaction.get("reactionmeddrapt"))
        if text:
            terms.append(text)
    return _unique_texts(terms)


def _extract_seriousness_flags(result: dict[str, Any]) -> list[str]:
    flags = [
        ("seriousnessdeath", "death"),
        ("seriousnesshospitalization", "hospitalization"),
        ("seriousnessdisabling", "disabling"),
        ("seriousnesslifethreatening", "life-threatening"),
        ("seriousnesscongenitalanomali", "congenital anomaly"),
        ("seriousnessother", "other serious event"),
    ]
    return [label for field_name, label in flags if _first_text(result.get(field_name)) == "1"]


def _extract_report_date(result: dict[str, Any]) -> date | None:
    for field_name in ("receivedate", "receiptdate", "transmissiondate"):
        parsed = _parse_date(result.get(field_name))
        if parsed is not None:
            return parsed
    return None


def _make_source_id(result: dict[str, Any]) -> str:
    report_id = _first_text(result.get("safetyreportid"))
    version = _first_text(result.get("safetyreportversion"))
    if report_id and version:
        return f"{report_id}-v{version}"
    if report_id:
        return report_id

    fingerprint = json.dumps(
        {
            "receivedate": _first_text(result.get("receivedate")),
            "receiptdate": _first_text(result.get("receiptdate")),
            "drug_names": _extract_drug_names(result),
            "reaction_terms": _extract_reaction_terms(result),
            "reporttype": _first_text(result.get("reporttype")),
            "primarysourcecountry": _first_text(result.get("primarysourcecountry")),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"openfda-{hashlib.sha1(fingerprint.encode('utf-8')).hexdigest()}"


def _make_title(drug_names: list[str], reaction_terms: list[str]) -> str:
    if drug_names and reaction_terms:
        return f"{drug_names[0]} adverse event report: {reaction_terms[0]}"
    if drug_names:
        return f"{drug_names[0]} adverse event report"
    if reaction_terms:
        return f"Adverse event report: {reaction_terms[0]}"
    return "openFDA adverse event report"


def _make_description(
    *,
    drug_names: list[str],
    reaction_terms: list[str],
    report_date: date | None,
    seriousness_flags: list[str],
    report_id: str,
) -> str:
    parts: list[str] = []
    if drug_names:
        parts.append(f"Reported drug(s): {', '.join(drug_names[:3])}.")
    if reaction_terms:
        parts.append(f"Reported reaction(s): {', '.join(reaction_terms[:5])}.")
    if seriousness_flags:
        parts.append(f"Seriousness flags: {', '.join(seriousness_flags)}.")
    if report_date is not None:
        parts.append(f"Report date: {report_date.isoformat()}.")
    parts.append(f"Report ID: {report_id}.")
    parts.append("This record describes a reported adverse event signal and does not establish causation.")
    return " ".join(parts)


def _normalize_result(result: dict[str, Any]) -> MonitorItem:
    report_date = _extract_report_date(result)
    drug_names = _extract_drug_names(result)
    reaction_terms = _extract_reaction_terms(result)
    report_id = _make_source_id(result)

    return MonitorItem(
        source="openFDA",
        title=_make_title(drug_names, reaction_terms),
        published_date=report_date,
        category="Drug Adverse Event",
        description=_make_description(
            drug_names=drug_names,
            reaction_terms=reaction_terms,
            report_date=report_date,
            seriousness_flags=_extract_seriousness_flags(result),
            report_id=report_id,
        ),
        url=OPENFDA_ENDPOINT_DOC_URL,
        source_id=report_id,
    )


class OpenFDAAdapter(MonitorSourceAdapter):
    """Fetch and normalize openFDA drug adverse event reports."""

    def fetch(
        self,
        topic: str,
        *,
        published_after: date | None = None,
        published_before: date | None = None,
        limit: int = 20,
    ) -> list[MonitorItem]:
        if limit <= 0:
            logger.debug("openFDA fetch requested with non-positive limit=%s", limit)
            return []

        normalized_topic = topic.strip()
        if not normalized_topic:
            logger.debug("openFDA fetch requested with an empty topic")
            return []

        logger.info(
            "Fetching openFDA adverse events for topic=%r limit=%s after=%s before=%s",
            normalized_topic,
            limit,
            published_after,
            published_before,
        )

        payload = _read_json(_build_request_url(normalized_topic, limit=limit))
        results = _extract_results(payload)

        items: list[MonitorItem] = []
        for result in results:
            item = _normalize_result(result)
            if not _passes_date_filter(
                item.published_date,
                published_after=published_after,
                published_before=published_before,
            ):
                continue
            items.append(item)

        sorted_items = sort_monitor_items_newest_first(items)
        logger.info(
            "openFDA fetch completed for topic=%r with %s item(s)",
            normalized_topic,
            len(sorted_items),
        )
        return sorted_items

"""ClinicalTrials.gov Monitor source adapter."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

from src.monitor import (
    MonitorItem,
    MonitorSourceAdapter,
    sort_monitor_items_newest_first,
)


logger = logging.getLogger(__name__)

CLINICAL_TRIALS_API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"
HTTP_TIMEOUT_SECONDS = 10.0
MAX_PAGE_SIZE = 1000


class ClinicalTrialsAdapterError(RuntimeError):
    """Raised when the ClinicalTrials.gov adapter cannot fetch or parse data."""

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
        for key in ("text", "name", "title", "description", "label", "value"):
            flattened = _flatten_text(value.get(key))
            if flattened:
                return flattened
        flattened_values: list[str] = []
        for nested_value in value.values():
            flattened_values.extend(_flatten_text(nested_value))
        return flattened_values
    if isinstance(value, (list, tuple, set)):
        flattened_values: list[str] = []
        for nested_value in value:
            flattened_values.extend(_flatten_text(nested_value))
        return flattened_values
    text = str(value).strip()
    return [text] if text else []


def _first_text(value: Any) -> str | None:
    flattened = _flatten_text(value)
    return flattened[0] if flattened else None


def _parse_date(value: Any) -> date | None:
    text = _first_text(value)
    if not text:
        return None

    for date_format in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        if date_format == "%Y":
            return date(parsed.year, 1, 1)
        if date_format == "%Y-%m":
            return date(parsed.year, parsed.month, 1)
        return parsed.date()

    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _build_request_url(
    topic: str,
    *,
    limit: int,
    page_token: str | None = None,
) -> str:
    params: dict[str, Any] = {
        "query.term": topic,
        "pageSize": min(max(limit, 1), MAX_PAGE_SIZE),
    }
    if page_token:
        params["pageToken"] = page_token
    return f"{CLINICAL_TRIALS_API_BASE_URL}?{parse.urlencode(params)}"


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
        raise ClinicalTrialsAdapterError(
            f"ClinicalTrials.gov returned HTTP {exc.code}",
            detail=detail,
        ) from exc
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov request failed",
            detail=str(exc),
        ) from exc

    if status != 200:
        raise ClinicalTrialsAdapterError(
            f"ClinicalTrials.gov returned HTTP {status}",
            detail=raw_body.decode("utf-8", errors="replace").strip() or None,
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov returned malformed JSON",
            detail=str(exc),
        ) from exc

    if not isinstance(payload, dict):
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov returned an unexpected response payload",
            detail=f"Expected a JSON object, got {type(payload).__name__}",
        )
    return payload


def _extract_studies(payload: dict[str, Any]) -> list[dict[str, Any]]:
    studies = payload.get("studies")
    if isinstance(studies, list):
        return [study for study in studies if isinstance(study, dict)]

    nested_studies = payload.get("study")
    if isinstance(nested_studies, list):
        return [study for study in nested_studies if isinstance(study, dict)]

    raise ClinicalTrialsAdapterError(
        "ClinicalTrials.gov returned an unexpected response structure",
        detail="Missing studies list in response payload",
    )


def _extract_next_page_token(payload: dict[str, Any]) -> str | None:
    token = payload.get("nextPageToken") or payload.get("next_page_token")
    return _first_text(token)


def _passes_date_filter(
    published_date: date | None,
    *,
    published_after: date | None,
    published_before: date | None,
) -> bool:
    if published_date is None:
        return published_after is None and published_before is None
    if published_after is not None and published_date < published_after:
        return False
    if published_before is not None and published_date > published_before:
        return False
    return True


def _build_description(protocol_section: dict[str, Any]) -> str:
    description_parts: list[str] = []

    summary = _first_text(_nested_get(protocol_section, "descriptionModule", "briefSummary"))
    if not summary:
        summary = _first_text(_nested_get(protocol_section, "descriptionModule", "detailedDescription"))
    if summary:
        description_parts.append(summary)

    status = _first_text(_nested_get(protocol_section, "statusModule", "overallStatus"))
    if status:
        description_parts.append(f"Status: {status.title()}")

    conditions = _flatten_text(_nested_get(protocol_section, "conditionsModule", "conditions"))
    if conditions:
        description_parts.append(f"Conditions: {', '.join(conditions[:5])}")

    interventions = _flatten_text(_nested_get(protocol_section, "armsInterventionsModule", "interventions"))
    if interventions:
        description_parts.append(f"Interventions: {', '.join(interventions[:5])}")

    phases = _flatten_text(_nested_get(protocol_section, "designModule", "phases"))
    if phases:
        description_parts.append(f"Phases: {', '.join(phases[:3])}")

    description = " ".join(description_parts).strip()
    return description or "ClinicalTrials.gov study record."


def _normalize_study(study: dict[str, Any]) -> MonitorItem:
    protocol_section = study.get("protocolSection")
    if not isinstance(protocol_section, dict):
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov study missing protocolSection",
            detail="Study payload did not contain a protocolSection object",
        )

    identification_module = protocol_section.get("identificationModule")
    if not isinstance(identification_module, dict):
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov study missing identificationModule",
            detail="Study payload did not contain an identificationModule object",
        )

    nct_id = _first_text(identification_module.get("nctId")) or _first_text(study.get("nctId"))
    if not nct_id:
        raise ClinicalTrialsAdapterError(
            "ClinicalTrials.gov study missing NCT identifier",
            detail="Unable to locate NCT ID in study payload",
        )

    title = _first_text(identification_module.get("officialTitle")) or _first_text(
        identification_module.get("briefTitle")
    )
    if not title:
        title = nct_id

    status_module = protocol_section.get("statusModule")
    if not isinstance(status_module, dict):
        status_module = {}

    last_update_date = _parse_date(_nested_get(status_module, "lastUpdatePostDateStruct", "date"))
    if last_update_date is None:
        last_update_date = _parse_date(_nested_get(status_module, "studyFirstPostDateStruct", "date"))

    study_type = _first_text(_nested_get(protocol_section, "designModule", "studyType"))
    overall_status = _first_text(_nested_get(protocol_section, "statusModule", "overallStatus"))
    category = f"{(study_type or 'unknown').upper()} | {(overall_status or 'unknown').upper()}"

    description = _build_description(protocol_section)
    url = CLINICAL_TRIALS_STUDY_URL.format(nct_id=nct_id)

    return MonitorItem(
        source="ClinicalTrials.gov",
        title=title,
        published_date=last_update_date,
        category=category,
        description=description,
        url=url,
        source_id=nct_id,
    )


class ClinicalTrialsGovAdapter(MonitorSourceAdapter):
    """Fetch and normalize ClinicalTrials.gov studies into Monitor items."""

    def fetch(
        self,
        topic: str,
        *,
        published_after: date | None = None,
        published_before: date | None = None,
        limit: int = 20,
    ) -> list[MonitorItem]:
        if limit <= 0:
            logger.debug("ClinicalTrials.gov fetch requested with non-positive limit=%s", limit)
            return []

        normalized_topic = topic.strip()
        if not normalized_topic:
            logger.debug("ClinicalTrials.gov fetch requested with an empty topic")
            return []

        logger.info(
            "Fetching ClinicalTrials.gov studies for topic=%r limit=%s after=%s before=%s",
            normalized_topic,
            limit,
            published_after,
            published_before,
        )

        items: list[MonitorItem] = []
        page_token: str | None = None
        target_limit = min(limit, MAX_PAGE_SIZE)

        while len(items) < target_limit:
            page_size = min(target_limit - len(items), MAX_PAGE_SIZE)
            url = _build_request_url(normalized_topic, limit=page_size, page_token=page_token)
            payload = _read_json(url)
            studies = _extract_studies(payload)

            for study in studies:
                item = _normalize_study(study)
                if not _passes_date_filter(
                    item.published_date,
                    published_after=published_after,
                    published_before=published_before,
                ):
                    continue
                items.append(item)
                if len(items) >= target_limit:
                    break

            page_token = _extract_next_page_token(payload)
            if not page_token or len(items) >= target_limit:
                break

        sorted_items = sort_monitor_items_newest_first(items)
        logger.info(
            "ClinicalTrials.gov fetch completed for topic=%r with %s item(s)",
            normalized_topic,
            len(sorted_items),
        )
        return sorted_items

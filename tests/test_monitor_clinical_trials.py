from __future__ import annotations

import io
import json
from datetime import date
from urllib import error as urlerror
from urllib import parse

import pytest

from src.monitor_sources import clinical_trials as ct_module


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_study(
    *,
    nct_id: str,
    official_title: str | None = None,
    brief_title: str | None = None,
    last_update_post_date: str | None = None,
    study_first_post_date: str | None = None,
    overall_status: str = "Recruiting",
    study_type: str = "Interventional",
    brief_summary: str | None = "Study summary text.",
    detailed_description: str | None = None,
    conditions: list[str] | None = None,
    interventions: list[str] | None = None,
    phases: list[str] | None = None,
) -> dict[str, object]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                **({"officialTitle": official_title} if official_title is not None else {}),
                **({"briefTitle": brief_title} if brief_title is not None else {}),
            },
            "statusModule": {
                "overallStatus": overall_status,
                **(
                    {"lastUpdatePostDateStruct": {"date": last_update_post_date}}
                    if last_update_post_date is not None
                    else {}
                ),
                **(
                    {"studyFirstPostDateStruct": {"date": study_first_post_date}}
                    if study_first_post_date is not None
                    else {}
                ),
            },
            "designModule": {
                "studyType": study_type,
                **({"phases": phases} if phases is not None else {}),
            },
            "descriptionModule": {
                **({"briefSummary": brief_summary} if brief_summary is not None else {}),
                **({"detailedDescription": detailed_description} if detailed_description is not None else {}),
            },
            "conditionsModule": {
                **({"conditions": conditions} if conditions is not None else {}),
            },
            "armsInterventionsModule": {
                **({"interventions": interventions} if interventions is not None else {}),
            },
        }
    }


def _make_payload(studies: list[dict[str, object]], *, next_page_token: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"studies": studies}
    if next_page_token is not None:
        payload["nextPageToken"] = next_page_token
    return payload


def test_fetch_constructs_expected_request_and_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request_obj, timeout):
        captured["url"] = request_obj.full_url
        captured["timeout"] = timeout
        return FakeResponse(
            _make_payload(
                [
                    _make_study(
                        nct_id="NCT01234567",
                        official_title="Official pharmacovigilance study title",
                        brief_title="Brief title fallback",
                        last_update_post_date="2026-08-10",
                        overall_status="Recruiting",
                        study_type="Interventional",
                        brief_summary="A concise summary of the study.",
                        conditions=["Parkinson Disease"],
                        interventions=["Drug A"],
                        phases=["Phase 2"],
                    )
                ]
            )
        )

    monkeypatch.setattr(ct_module.request, "urlopen", fake_urlopen)

    adapter = ct_module.ClinicalTrialsGovAdapter()
    items = adapter.fetch("Parkinson", limit=5)

    assert len(items) == 1
    assert captured["timeout"] == ct_module.HTTP_TIMEOUT_SECONDS

    parsed_url = parse.urlparse(str(captured["url"]))
    query_params = parse.parse_qs(parsed_url.query)
    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == "clinicaltrials.gov"
    assert parsed_url.path == "/api/v2/studies"
    assert query_params["query.term"] == ["Parkinson"]
    assert query_params["pageSize"] == ["5"]

    item = items[0]
    assert item.source == "ClinicalTrials.gov"
    assert item.source_id == "NCT01234567"
    assert item.title == "Official pharmacovigilance study title"
    assert item.published_date == date(2026, 8, 10)
    assert item.category == "INTERVENTIONAL | RECRUITING"
    assert item.url == "https://clinicaltrials.gov/study/NCT01234567"
    assert "A concise summary of the study." in item.description
    assert "Conditions: Parkinson Disease" in item.description


def test_fetch_applies_date_filter_and_handles_missing_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _make_payload(
                [
                    _make_study(
                        nct_id="NCT00000001",
                        brief_title="Brief title fallback",
                        study_first_post_date="2026-08-09",
                        overall_status="Recruiting",
                        study_type="Observational",
                        brief_summary=None,
                        conditions=None,
                        interventions=None,
                    ),
                    _make_study(
                        nct_id="NCT00000002",
                        official_title="Too old study",
                        last_update_post_date="2026-07-01",
                        overall_status="Completed",
                        study_type="Interventional",
                        brief_summary="Old study summary.",
                    ),
                    _make_study(
                        nct_id="NCT00000003",
                        official_title="Missing date study",
                        overall_status="Recruiting",
                        study_type="Interventional",
                        brief_summary="No published date available.",
                    ),
                ]
            )
        )

    monkeypatch.setattr(ct_module.request, "urlopen", fake_urlopen)

    adapter = ct_module.ClinicalTrialsGovAdapter()
    items = adapter.fetch(
        "Parkinson",
        published_after=date(2026, 8, 1),
        published_before=date(2026, 8, 31),
        limit=10,
    )

    assert [item.source_id for item in items] == ["NCT00000001"]
    assert items[0].title == "Brief title fallback"
    assert items[0].published_date == date(2026, 8, 9)
    assert items[0].category == "OBSERVATIONAL | RECRUITING"
    assert "Status: Recruiting" in items[0].description


def test_fetch_returns_missing_date_items_when_no_filters_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _make_payload(
                [
                    _make_study(
                        nct_id="NCT11111111",
                        official_title="Study with no date",
                        overall_status="Recruiting",
                        study_type="Interventional",
                        brief_summary="Summary only.",
                    )
                ]
            )
        )

    monkeypatch.setattr(ct_module.request, "urlopen", fake_urlopen)

    adapter = ct_module.ClinicalTrialsGovAdapter()
    items = adapter.fetch("Parkinson", limit=1)

    assert len(items) == 1
    assert items[0].published_date is None


def test_fetch_wraps_timeout_and_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_urlopen(request_obj, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ct_module.request, "urlopen", timeout_urlopen)

    adapter = ct_module.ClinicalTrialsGovAdapter()

    with pytest.raises(ct_module.ClinicalTrialsAdapterError, match="request failed"):
        adapter.fetch("Parkinson", limit=1)

    def http_error_urlopen(request_obj, timeout):
        raise urlerror.HTTPError(
            request_obj.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"temporary outage"),
        )

    monkeypatch.setattr(ct_module.request, "urlopen", http_error_urlopen)

    with pytest.raises(ct_module.ClinicalTrialsAdapterError, match="HTTP 503"):
        adapter.fetch("Parkinson", limit=1)


def test_fetch_wraps_malformed_and_unexpected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def malformed_json_urlopen(request_obj, timeout):
        return FakeResponse({}, status=200)

    monkeypatch.setattr(ct_module.request, "urlopen", malformed_json_urlopen)

    adapter = ct_module.ClinicalTrialsGovAdapter()

    with pytest.raises(ct_module.ClinicalTrialsAdapterError, match="unexpected response structure"):
        adapter.fetch("Parkinson", limit=1)

    class MalformedJSONResponse(FakeResponse):
        def __init__(self) -> None:
            self._body = b"{not valid json"
            self.status = 200

    def invalid_json_urlopen(request_obj, timeout):
        return MalformedJSONResponse()

    monkeypatch.setattr(ct_module.request, "urlopen", invalid_json_urlopen)

    with pytest.raises(ct_module.ClinicalTrialsAdapterError, match="malformed JSON"):
        adapter.fetch("Parkinson", limit=1)


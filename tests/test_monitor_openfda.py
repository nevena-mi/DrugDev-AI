from __future__ import annotations

import io
import json
from datetime import date
from urllib import error as urlerror
from urllib import parse

import pytest

from src.monitor_sources import openfda as openfda_module


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


def _make_report(
    *,
    safetyreportid: str,
    safetyreportversion: str | None = None,
    receivedate: str | None = None,
    receiptdate: str | None = None,
    transmissiondate: str | None = None,
    drug_names: list[str] | None = None,
    reaction_terms: list[str] | None = None,
    serious_flags: dict[str, str] | None = None,
    reporttype: str | None = "1",
    primarysourcecountry: str | None = "US",
) -> dict[str, object]:
    drugs = []
    for drug_name in drug_names or []:
        drugs.append(
            {
                "medicinalproduct": drug_name,
                "openfda": {
                    "brand_name": [drug_name],
                    "generic_name": [drug_name.lower()],
                    "substance_name": [drug_name.upper()],
                },
            }
        )

    reactions = []
    for reaction_term in reaction_terms or []:
        reactions.append({"reactionmeddrapt": reaction_term})

    report: dict[str, object] = {
        "safetyreportid": safetyreportid,
        "patient": {
            "drug": drugs,
            "reaction": reactions,
        },
    }
    if safetyreportversion is not None:
        report["safetyreportversion"] = safetyreportversion
    if receivedate is not None:
        report["receivedate"] = receivedate
    if receiptdate is not None:
        report["receiptdate"] = receiptdate
    if transmissiondate is not None:
        report["transmissiondate"] = transmissiondate
    if reporttype is not None:
        report["reporttype"] = reporttype
    if primarysourcecountry is not None:
        report["primarysourcecountry"] = primarysourcecountry
    if serious_flags:
        report.update(serious_flags)
    return report


def _make_payload(reports: list[dict[str, object]]) -> dict[str, object]:
    return {"results": reports}


def test_fetch_constructs_expected_request_and_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request_obj, timeout):
        captured["url"] = request_obj.full_url
        captured["timeout"] = timeout
        return FakeResponse(
            _make_payload(
                [
                    _make_report(
                        safetyreportid="111111",
                        safetyreportversion="2",
                        receivedate="20260810",
                        drug_names=["Metformin"],
                        reaction_terms=["Nausea"],
                        serious_flags={"seriousnesshospitalization": "1"},
                    )
                ]
            )
        )

    monkeypatch.setattr(openfda_module.request, "urlopen", fake_urlopen)

    adapter = openfda_module.OpenFDAAdapter()
    items = adapter.fetch("Metformin", limit=5)

    assert len(items) == 1
    assert captured["timeout"] == openfda_module.HTTP_TIMEOUT_SECONDS

    parsed_url = parse.urlparse(str(captured["url"]))
    query_params = parse.parse_qs(parsed_url.query)
    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == "api.fda.gov"
    assert parsed_url.path == "/drug/event.json"
    assert query_params["limit"] == ["5"]
    assert query_params["sort"] == ["receivedate:desc"]
    assert query_params["search"] == [
        '(patient.drug.medicinalproduct:"Metformin" OR patient.drug.openfda.brand_name:"Metformin" OR patient.drug.openfda.generic_name:"Metformin" OR patient.reaction.reactionmeddrapt:"Metformin")'
    ]

    item = items[0]
    assert item.source == "openFDA"
    assert item.source_id == "111111-v2"
    assert item.title == "Metformin adverse event report: Nausea"
    assert item.published_date == date(2026, 8, 10)
    assert item.category == "Drug Adverse Event"
    assert item.url == openfda_module.OPENFDA_ENDPOINT_DOC_URL
    assert "Reported drug(s): Metformin." in item.description
    assert "Reported reaction(s): Nausea." in item.description
    assert "does not establish causation" in item.description.lower()


def test_fetch_includes_api_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request_obj, timeout):
        captured["url"] = request_obj.full_url
        return FakeResponse(_make_payload([_make_report(safetyreportid="222222", drug_names=["DrugA"])]))

    monkeypatch.setattr(openfda_module, "OPENFDA_API_KEY", "test-key")
    monkeypatch.setattr(openfda_module.request, "urlopen", fake_urlopen)

    adapter = openfda_module.OpenFDAAdapter()
    adapter.fetch("DrugA", limit=1)

    query_params = parse.parse_qs(parse.urlparse(str(captured["url"])).query)
    assert query_params["api_key"] == ["test-key"]


def test_fetch_applies_date_filter_and_handles_missing_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _make_payload(
                [
                    _make_report(
                        safetyreportid="333333",
                        receivedate="20260810",
                        drug_names=["DrugA"],
                        reaction_terms=["Headache"],
                    ),
                    _make_report(
                        safetyreportid="444444",
                        receivedate="20260701",
                        drug_names=["DrugB"],
                        reaction_terms=["Dizziness"],
                    ),
                    _make_report(
                        safetyreportid="555555",
                        drug_names=["DrugC"],
                        reaction_terms=["Fatigue"],
                    ),
                ]
            )
        )

    monkeypatch.setattr(openfda_module.request, "urlopen", fake_urlopen)

    adapter = openfda_module.OpenFDAAdapter()
    items = adapter.fetch(
        "Drug",
        published_after=date(2026, 8, 1),
        published_before=date(2026, 8, 31),
        limit=10,
    )

    assert [item.source_id for item in items] == ["333333"]
    assert items[0].published_date == date(2026, 8, 10)
    assert items[0].title == "DrugA adverse event report: Headache"


def test_fetch_returns_missing_date_items_when_no_filters_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(_make_payload([_make_report(safetyreportid="666666", drug_names=["DrugD"])]))

    monkeypatch.setattr(openfda_module.request, "urlopen", fake_urlopen)

    adapter = openfda_module.OpenFDAAdapter()
    items = adapter.fetch("DrugD", limit=1)

    assert len(items) == 1
    assert items[0].published_date is None


def test_fetch_wraps_timeout_and_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_urlopen(request_obj, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(openfda_module.request, "urlopen", timeout_urlopen)

    adapter = openfda_module.OpenFDAAdapter()

    with pytest.raises(openfda_module.OpenFDAAdapterError, match="request failed"):
        adapter.fetch("DrugA", limit=1)

    def http_error_urlopen(request_obj, timeout):
        raise urlerror.HTTPError(
            request_obj.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b"rate limited"),
        )

    monkeypatch.setattr(openfda_module.request, "urlopen", http_error_urlopen)

    with pytest.raises(openfda_module.OpenFDAAdapterError, match="HTTP 429"):
        adapter.fetch("DrugA", limit=1)


def test_fetch_wraps_malformed_and_unexpected_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_structure_urlopen(request_obj, timeout):
        return FakeResponse({"meta": {}}, status=200)

    monkeypatch.setattr(openfda_module.request, "urlopen", unexpected_structure_urlopen)

    adapter = openfda_module.OpenFDAAdapter()

    with pytest.raises(openfda_module.OpenFDAAdapterError, match="unexpected response structure"):
        adapter.fetch("DrugA", limit=1)

    class MalformedJSONResponse(FakeResponse):
        def __init__(self) -> None:
            self._body = b"{not valid json"
            self.status = 200

    def invalid_json_urlopen(request_obj, timeout):
        return MalformedJSONResponse()

    monkeypatch.setattr(openfda_module.request, "urlopen", invalid_json_urlopen)

    with pytest.raises(openfda_module.OpenFDAAdapterError, match="malformed JSON"):
        adapter.fetch("DrugA", limit=1)


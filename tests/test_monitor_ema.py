from __future__ import annotations

import io
from datetime import date
from urllib import error as urlerror
from urllib import parse

import pytest

from src.monitor_sources import ema as ema_module


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _rss_item(
    *,
    title: str | None = None,
    link: str | None = None,
    guid: str | None = None,
    pub_date: str | None = None,
    description: str | None = None,
    summary: str | None = None,
) -> str:
    parts = ["<item>"]
    if title is not None:
        parts.append(f"<title>{title}</title>")
    if link is not None:
        parts.append(f"<link>{link}</link>")
    if description is not None:
        parts.append(f"<description>{description}</description>")
    if summary is not None:
        parts.append(f"<summary>{summary}</summary>")
    if guid is not None:
        parts.append(f"<guid>{guid}</guid>")
    if pub_date is not None:
        parts.append(f"<pubDate>{pub_date}</pubDate>")
    parts.append("</item>")
    return "".join(parts)


def _rss_feed(items: list[str]) -> bytes:
    return (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        "<rss version=\"2.0\"><channel>"
        "<title>News and press releases</title>"
        "<link>https://www.ema.europa.eu/en/homepage</link>"
        "<description>News and press releases</description>"
        + "".join(items)
        + "</channel></rss>"
    ).encode("utf-8")


def test_fetch_constructs_expected_request_and_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request_obj, timeout):
        captured["url"] = request_obj.full_url
        captured["timeout"] = timeout
        return FakeResponse(
            _rss_feed(
                [
                    _rss_item(
                        title="EMA announces new guidance",
                        link="https://www.ema.europa.eu/en/news/ema-announces-new-guidance",
                        guid="ema-guid-001",
                        pub_date="Thu, 30 Jul 2026 17:27:00 +0200",
                        description="EMA announces new guidance on medicines evaluation.",
                    )
                ]
            )
        )

    monkeypatch.setattr(ema_module.request, "urlopen", fake_urlopen)

    adapter = ema_module.EMAAdapter()
    items = adapter.fetch("guidance", limit=5)

    assert len(items) == 1
    assert captured["timeout"] == ema_module.HTTP_TIMEOUT_SECONDS

    parsed_url = parse.urlparse(str(captured["url"]))
    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == "www.ema.europa.eu"
    assert parsed_url.path == "/en/news.xml"
    assert parsed_url.query == ""

    item = items[0]
    assert item.source == "EMA"
    assert item.source_id == "ema-guid-001"
    assert item.title == "EMA announces new guidance"
    assert item.published_date == date(2026, 7, 30)
    assert item.category == "EMA News"
    assert item.description == "EMA announces new guidance on medicines evaluation."
    assert item.url == "https://www.ema.europa.eu/en/news/ema-announces-new-guidance"


def test_fetch_is_case_insensitive_and_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _rss_feed(
                [
                    _rss_item(
                        title="EMA and AI in medicines regulation",
                        link="https://www.ema.europa.eu/en/news/ema-ai",
                        guid="ema-ai-001",
                        pub_date="Fri, 01 Aug 2026 10:00:00 +0200",
                        description="EMA news on AI.",
                    ),
                    _rss_item(
                        title="Clinical trials update",
                        link="https://www.ema.europa.eu/en/news/clinical-trials-update",
                        guid="ema-ct-001",
                        pub_date="Thu, 31 Jul 2026 10:00:00 +0200",
                        description="Clinical trials and regulatory matters.",
                    ),
                ]
            )
        )

    monkeypatch.setattr(ema_module.request, "urlopen", fake_urlopen)

    adapter = ema_module.EMAAdapter()
    items = adapter.fetch("AI", limit=1)

    assert len(items) == 1
    assert items[0].source_id == "ema-ai-001"


def test_fetch_applies_date_filter_and_handles_missing_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _rss_feed(
                [
                    _rss_item(
                        title="EMA item with date",
                        link="https://www.ema.europa.eu/en/news/with-date",
                        guid="ema-date-001",
                        pub_date="Fri, 01 Aug 2026 10:00:00 +0200",
                        description="A dated EMA item.",
                    ),
                    _rss_item(
                        title="EMA item too old",
                        link="https://www.ema.europa.eu/en/news/too-old",
                        guid="ema-old-001",
                        pub_date="Mon, 01 Jun 2026 10:00:00 +0200",
                        description="An older EMA item.",
                    ),
                    _rss_item(
                        title="EMA item missing date",
                        link="https://www.ema.europa.eu/en/news/missing-date",
                        guid="ema-missing-001",
                        description="Missing publication date.",
                    ),
                    _rss_item(
                        title="EMA item missing guid and description",
                        link="https://www.ema.europa.eu/en/news/fallback-id",
                        pub_date="Fri, 01 Aug 2026 10:00:00 +0200",
                    ),
                ]
            )
        )

    monkeypatch.setattr(ema_module.request, "urlopen", fake_urlopen)

    adapter = ema_module.EMAAdapter()
    items = adapter.fetch(
        "EMA item",
        published_after=date(2026, 7, 1),
        published_before=date(2026, 8, 31),
        limit=10,
    )

    assert [item.source_id for item in items] == [
        "ema-date-001",
        "https://www.ema.europa.eu/en/news/fallback-id",
    ]
    assert items[0].published_date == date(2026, 8, 1)
    assert items[1].description == "EMA item missing guid and description"
    assert items[1].source_id == "https://www.ema.europa.eu/en/news/fallback-id"


def test_fetch_returns_missing_date_items_when_no_filters_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request_obj, timeout):
        return FakeResponse(
            _rss_feed(
                [
                    _rss_item(
                        title="EMA item without date",
                        link="https://www.ema.europa.eu/en/news/no-date",
                        guid="ema-no-date-001",
                        description="Item without pubDate.",
                    )
                ]
            )
        )

    monkeypatch.setattr(ema_module.request, "urlopen", fake_urlopen)

    adapter = ema_module.EMAAdapter()
    items = adapter.fetch("EMA item", limit=1)

    assert len(items) == 1
    assert items[0].published_date is None


def test_fetch_wraps_timeout_and_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_urlopen(request_obj, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ema_module.request, "urlopen", timeout_urlopen)

    adapter = ema_module.EMAAdapter()

    with pytest.raises(ema_module.EMAAdapterError, match="request failed"):
        adapter.fetch("AI", limit=1)

    def http_error_urlopen(request_obj, timeout):
        raise urlerror.HTTPError(
            request_obj.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"maintenance"),
        )

    monkeypatch.setattr(ema_module.request, "urlopen", http_error_urlopen)

    with pytest.raises(ema_module.EMAAdapterError, match="HTTP 503"):
        adapter.fetch("AI", limit=1)


def test_fetch_wraps_malformed_and_unexpected_xml(monkeypatch: pytest.MonkeyPatch) -> None:
    def malformed_xml_urlopen(request_obj, timeout):
        return FakeResponse(b"<rss><channel><item></channel></rss>")

    monkeypatch.setattr(ema_module.request, "urlopen", malformed_xml_urlopen)

    adapter = ema_module.EMAAdapter()

    with pytest.raises(ema_module.EMAAdapterError, match="malformed XML"):
        adapter.fetch("AI", limit=1)

    def unexpected_structure_urlopen(request_obj, timeout):
        return FakeResponse(b"<feed><entry></entry></feed>")

    monkeypatch.setattr(ema_module.request, "urlopen", unexpected_structure_urlopen)

    with pytest.raises(ema_module.EMAAdapterError, match="unexpected feed structure"):
        adapter.fetch("AI", limit=1)

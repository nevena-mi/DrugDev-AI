"""EMA RSS Monitor source adapter."""

from __future__ import annotations

import html
import logging
import re
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any
from urllib import error as urlerror
from urllib import request
import xml.etree.ElementTree as ET

from src.monitor import MonitorItem, MonitorSourceAdapter


logger = logging.getLogger(__name__)

EMA_RSS_FEED_URL = "https://www.ema.europa.eu/en/news.xml"
EMA_RSS_FEED_NAME = "News and press releases"
HTTP_TIMEOUT_SECONDS = 10.0


class EMAAdapterError(RuntimeError):
    """Raised when the EMA RSS adapter cannot fetch or parse feed data."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _strip_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    return parsed.date()


def _normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic).strip().lower()


def _matches_topic(topic: str, *, title: str, description: str) -> bool:
    normalized_topic = _normalize_topic(topic)
    if not normalized_topic:
        return False
    searchable_text = f"{title} {description}".lower()
    return normalized_topic in searchable_text


def _build_request(url: str) -> request.Request:
    return request.Request(
        url,
        headers={
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            "User-Agent": "DrugDev-AI-Monitor/1.0",
        },
    )


def _read_xml(url: str) -> bytes:
    try:
        with request.urlopen(_build_request(url), timeout=HTTP_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", response.getcode())
            body = response.read()
    except urlerror.HTTPError as exc:
        detail = None
        if exc.fp is not None:
            try:
                detail = exc.fp.read().decode("utf-8", errors="replace").strip() or None
            except Exception:  # pragma: no cover - defensive
                detail = None
        raise EMAAdapterError(
            f"EMA returned HTTP {exc.code}",
            detail=detail,
        ) from exc
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise EMAAdapterError(
            "EMA request failed",
            detail=str(exc),
        ) from exc

    if status != 200:
        raise EMAAdapterError(
            f"EMA returned HTTP {status}",
            detail=body.decode("utf-8", errors="replace").strip() or None,
        )
    return body


def _find_text(item: ET.Element, tag_name: str) -> str:
    for child in item:
        local_name = child.tag.split("}", 1)[-1]
        if local_name == tag_name:
            return _strip_text(child.text)
    return ""


def _extract_item_summary(item: ET.Element) -> str:
    description = _find_text(item, "description")
    if description:
        return description
    summary = _find_text(item, "summary")
    if summary:
        return summary
    return ""


def _extract_item_guid(item: ET.Element) -> str:
    guid = _find_text(item, "guid")
    if guid:
        return guid
    link = _find_text(item, "link")
    if link:
        return link
    title = _find_text(item, "title")
    if title:
        return title
    raise EMAAdapterError(
        "EMA feed item missing a stable identifier",
        detail="Could not determine guid, link, or title for RSS item",
    )


def _extract_item_url(item: ET.Element) -> str:
    link = _find_text(item, "link")
    if link:
        return link
    guid = _find_text(item, "guid")
    if guid:
        return guid
    raise EMAAdapterError(
        "EMA feed item missing a URL",
        detail="Could not determine an official EMA item URL from the RSS item",
    )


def _extract_item_date(item: ET.Element) -> date | None:
    for tag_name in ("pubDate", "date"):
        published_date = _parse_date(_find_text(item, tag_name))
        if published_date is not None:
            return published_date
    return None


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


def _normalize_item(item: ET.Element) -> MonitorItem:
    title = _find_text(item, "title")
    if not title:
        raise EMAAdapterError(
            "EMA feed item missing a title",
            detail="RSS item did not include a title",
        )

    url = _extract_item_url(item)
    guid = _extract_item_guid(item)
    published_date = _extract_item_date(item)
    description = _extract_item_summary(item) or title

    return MonitorItem(
        source="EMA",
        title=title,
        published_date=published_date,
        category="EMA News",
        description=description,
        url=url,
        source_id=guid,
    )


def _parse_feed(xml_bytes: bytes) -> list[ET.Element]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise EMAAdapterError(
            "EMA returned malformed XML",
            detail=str(exc),
        ) from exc

    root_name = root.tag.split("}", 1)[-1]
    if root_name != "rss":
        raise EMAAdapterError(
            "EMA returned an unexpected feed structure",
            detail=f"Expected rss root element, got {root_name!r}",
        )

    channel = None
    for child in root:
        if child.tag.split("}", 1)[-1] == "channel":
            channel = child
            break
    if channel is None:
        raise EMAAdapterError(
            "EMA returned an unexpected feed structure",
            detail="Missing channel element in RSS feed",
        )

    items: list[ET.Element] = []
    for child in channel:
        if child.tag.split("}", 1)[-1] == "item":
            items.append(child)
    return items


class EMAAdapter(MonitorSourceAdapter):
    """Fetch and normalize EMA news items from the official RSS feed."""

    def fetch(
        self,
        topic: str,
        *,
        published_after: date | None = None,
        published_before: date | None = None,
        limit: int = 20,
    ) -> list[MonitorItem]:
        if limit <= 0:
            logger.debug("EMA fetch requested with non-positive limit=%s", limit)
            return []

        normalized_topic = topic.strip()
        if not normalized_topic:
            logger.debug("EMA fetch requested with an empty topic")
            return []

        logger.info(
            "Fetching EMA RSS feed for topic=%r limit=%s after=%s before=%s",
            normalized_topic,
            limit,
            published_after,
            published_before,
        )

        xml_bytes = _read_xml(EMA_RSS_FEED_URL)
        rss_items = _parse_feed(xml_bytes)

        matching_items: list[MonitorItem] = []
        for item in rss_items:
            normalized_item = _normalize_item(item)
            if not _matches_topic(
                normalized_topic,
                title=normalized_item.title,
                description=normalized_item.description,
            ):
                continue
            if not _passes_date_filter(
                normalized_item.published_date,
                published_after=published_after,
                published_before=published_before,
            ):
                continue
            matching_items.append(normalized_item)
            if len(matching_items) >= limit:
                break

        logger.info(
            "EMA fetch completed for topic=%r with %s item(s)",
            normalized_topic,
            len(matching_items),
        )
        return matching_items

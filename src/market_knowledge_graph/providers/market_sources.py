from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from ..models import Document


JsonFetcher = Callable[[str], dict]


def _json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "MarketGraphResearchBot/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(2_000_000)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("market data API returned a non-object response")
    return value


class AlphaVantageCollector:
    """Turn explicitly requested near-real-time stock quotes into graph documents."""

    def __init__(self, api_key: str, timeout: float = 10, fetcher: JsonFetcher | None = None) -> None:
        if not api_key.strip() or timeout <= 0:
            raise ValueError("Alpha Vantage API key and positive timeout are required")
        self.api_key, self.timeout, self.fetcher = api_key, timeout, fetcher

    def collect(self, symbols: Iterable[str]) -> list[Document]:
        documents: list[Document] = []
        for raw in list(symbols)[:25]:
            symbol = raw.strip().upper()
            if not symbol or not all(character.isalnum() or character in {".", "-"} for character in symbol):
                raise ValueError(f"invalid market symbol: {raw}")
            query = urllib.parse.urlencode({"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self.api_key})
            payload = self.fetcher(symbol) if self.fetcher else _json(f"https://www.alphavantage.co/query?{query}", self.timeout)
            quote = payload.get("Global Quote", payload)
            if not isinstance(quote, dict) or not quote:
                raise ValueError(f"no quote returned for {symbol}")
            price = quote.get("05. price") or quote.get("price")
            change = quote.get("10. change percent") or quote.get("change_percent") or "unknown"
            volume = quote.get("06. volume") or quote.get("volume") or "unknown"
            latest = str(quote.get("07. latest trading day") or quote.get("latest_trading_day") or datetime.now(timezone.utc).date())
            documents.append(Document(
                f"quote-{symbol.lower()}-{latest}",
                f"{symbol} market quote",
                f"{symbol} traded at {price}; daily change was {change}; reported volume was {volume} on {latest}.",
                "https://www.alphavantage.co/",
                latest,
            ))
        return documents


class GoogleSearchCollector:
    """Collect bounded Google Programmable Search snippets with source URLs."""

    def __init__(self, api_key: str, search_engine_id: str, timeout: float = 10, max_items: int = 10, fetcher: JsonFetcher | None = None) -> None:
        if not api_key.strip() or not search_engine_id.strip() or not 1 <= max_items <= 10:
            raise ValueError("Google API key, search engine id, and max_items 1–10 are required")
        self.api_key, self.search_engine_id, self.timeout, self.max_items, self.fetcher = api_key, search_engine_id, timeout, max_items, fetcher

    def collect(self, query: str) -> list[Document]:
        if not query.strip() or len(query) > 300:
            raise ValueError("search query must contain 1–300 characters")
        params = urllib.parse.urlencode({"key": self.api_key, "cx": self.search_engine_id, "q": query, "num": self.max_items})
        payload = self.fetcher(query) if self.fetcher else _json(f"https://www.googleapis.com/customsearch/v1?{params}", self.timeout)
        documents = []
        for item in payload.get("items", [])[: self.max_items]:
            if not isinstance(item, dict) or not item.get("link") or not item.get("snippet"):
                continue
            link = str(item["link"])
            identifier = hashlib.sha1(link.encode()).hexdigest()[:16]
            documents.append(Document(identifier, str(item.get("title") or "Search result"), str(item["snippet"]), link, "search-current"))
        return documents


class WebArticleCollector:
    """Bounded opt-in article extraction; callers must respect robots.txt and terms."""

    def __init__(self, timeout: float = 10, max_chars: int = 12_000, fetcher: Callable[[str], tuple[str, str]] | None = None) -> None:
        if timeout <= 0 or not 500 <= max_chars <= 100_000:
            raise ValueError("web timeout or text bound is invalid")
        self.timeout, self.max_chars, self.fetcher = timeout, max_chars, fetcher

    @staticmethod
    def _validate_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("web source must be an unauthenticated HTTP(S) URL")
        if parsed.hostname.lower() == "localhost":
            raise ValueError("local web sources are not allowed")
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError("private or reserved web sources are not allowed")
        return url

    def _fetch(self, url: str) -> tuple[str, str]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("Install web collection with `pip install -r requirements-data.txt`") from exc
        hostname = urllib.parse.urlparse(url).hostname
        if not hostname:
            raise ValueError("web source hostname is missing")
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        except (OSError, ValueError) as exc:
            raise ValueError("web source hostname could not be resolved safely") from exc
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("web source resolved to a private or reserved address")

        response = requests.get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": "MarketGraphResearchBot/1.0"},
            stream=True,
            allow_redirects=False,
        )
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError("redirecting web sources are rejected; provide the final public URL")
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("web source must return HTML content")
        content = bytearray()
        for chunk in response.iter_content(65_536):
            content.extend(chunk)
            if len(content) > 2_000_000:
                raise ValueError("web source exceeded the 2 MB download bound")
        soup = BeautifulSoup(bytes(content), "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else "Web article"
        return title, soup.get_text(" ", strip=True)

    def collect(self, urls: Iterable[str]) -> list[Document]:
        documents = []
        for url in list(urls)[:25]:
            safe_url = self._validate_url(url)
            title, text = (self.fetcher or self._fetch)(safe_url)
            text = " ".join(text.split())[: self.max_chars]
            if not text:
                continue
            identifier = hashlib.sha1(safe_url.encode()).hexdigest()[:16]
            documents.append(Document(identifier, title.strip() or "Web article", text, safe_url, "web-current"))
        return documents

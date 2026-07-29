from __future__ import annotations

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET

from ..models import Document


class RSSCollector:
    """Explicit opt-in RSS collector with bounded downloads; verify source terms first."""

    def __init__(self, timeout: float = 10, max_items: int = 25, user_agent: str = "MarketGraphResearchBot/1.0") -> None:
        if timeout <= 0 or max_items < 1 or max_items > 100:
            raise ValueError("RSS timeout must be positive and max_items must be between 1 and 100")
        if not user_agent.strip():
            raise ValueError("RSS user_agent is required")
        self.timeout, self.max_items, self.user_agent = timeout, max_items, user_agent

    def collect(self, url: str) -> list[Document]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read(2_000_000)
        root = ET.fromstring(payload)
        documents = []
        for item in root.findall(".//item")[: self.max_items]:
            title = (item.findtext("title") or "Untitled").strip()
            link = (item.findtext("link") or url).strip()
            description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            description = re.sub(r"\s+", " ", description).strip()
            published = (item.findtext("pubDate") or "unknown").strip()
            identifier = hashlib.sha1(link.encode()).hexdigest()[:16]
            if description:
                documents.append(Document(identifier, title, description, link, published))
        return documents

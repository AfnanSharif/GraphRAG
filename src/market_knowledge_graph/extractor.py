from __future__ import annotations

import re

from .graph import MarketGraph
from .models import Document

NAME = r"[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4}"
SECTORS = {
    "technology", "financial services", "energy", "healthcare", "consumer goods", "industrials",
    "telecommunications", "real estate", "utilities", "materials", "cloud infrastructure", "banking"
}


class EntityRelationExtractor:
    """Transparent pattern extractor intended as a baseline before model NER."""

    patterns = [
        # These patterns are intentionally case-sensitive: company tokens begin
        # with capitals, so a target stops cleanly before phrases such as
        # "to modernize" or "under an agreement".
        (re.compile(rf"(?P<a>{NAME})\s+(?:partnered|partners|is partnering)\s+with\s+(?P<b>{NAME})"), "partners_with"),
        (re.compile(rf"(?P<a>{NAME})\s+(?:acquired|acquires|is acquiring)\s+(?P<b>{NAME})"), "acquired"),
        (re.compile(rf"(?P<a>{NAME})\s+(?:supplies|will supply|is supplying)\s+(?P<b>{NAME})"), "supplies"),
        (re.compile(rf"(?P<a>{NAME})\s+(?:invested|invests|is investing)\s+in\s+(?P<b>{NAME})"), "invested_in"),
        (re.compile(rf"(?P<a>{NAME})\s+(?:competes|is competing)\s+with\s+(?P<b>{NAME})"), "competes_with"),
    ]

    def extract_into(self, graph: MarketGraph, document: Document) -> None:
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", document.text) if piece.strip()]
        for sentence in sentences:
            self._sectors(graph, document, sentence)
            for pattern, relation_kind in self.patterns:
                for match in pattern.finditer(sentence):
                    left = graph.add_entity(self._clean_name(match.group("a")), "company")
                    right = graph.add_entity(self._clean_name(match.group("b")), "company")
                    graph.add_relation(left, right, relation_kind, document.id, sentence)
            self._trend(graph, document, sentence)

    @staticmethod
    def _clean_name(name: str) -> str:
        prefixes = ("Meanwhile ", "Separately ", "Analysts said ", "The ")
        result = name.strip(" ,.")
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix) :]
        return result

    def _sectors(self, graph: MarketGraph, document: Document, sentence: str) -> None:
        pattern = re.compile(rf"(?P<company>{NAME})\s+(?:operates|is active)\s+in\s+(?:the\s+)?(?P<sector>[A-Za-z ]+?)(?:\s+sector)?[.,]", re.I)
        for match in pattern.finditer(sentence):
            raw_sector = match.group("sector").strip().lower()
            sector = next((value for value in SECTORS if raw_sector == value or raw_sector.endswith(value)), raw_sector)
            company = graph.add_entity(self._clean_name(match.group("company")), "company")
            sector_entity = graph.add_entity(sector.title(), "sector")
            graph.add_relation(company, sector_entity, "operates_in", document.id, sentence)

    @staticmethod
    def _trend(graph: MarketGraph, document: Document, sentence: str) -> None:
        pattern = re.compile(rf"(?P<company>{NAME})\s+(?:reported|announced)\s+(?P<trend>(?:rising|falling|strong|weak|higher|lower)\s+[a-z][a-z -]{{2,45}}?)(?:[.,]|\s+(?:while|as|after)\s)", re.I)
        for match in pattern.finditer(sentence):
            company = graph.add_entity(match.group("company").strip(), "company")
            trend = graph.add_entity(match.group("trend").strip().title(), "trend")
            graph.add_relation(company, trend, "reported_trend", document.id, sentence)

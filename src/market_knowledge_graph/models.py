from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    published_at: str


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    kind: str
    document_id: str
    evidence: str


@dataclass(frozen=True)
class Evidence:
    document_id: str
    title: str
    source: str
    published_at: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    graph_facts: tuple[str, ...]
    sentiment: str
    sentiment_score: float
    mode: str = "local-graph-rag"
    cautions: tuple[str, ...] = field(default_factory=lambda: ("This is source-grounded research assistance, not financial advice.",))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

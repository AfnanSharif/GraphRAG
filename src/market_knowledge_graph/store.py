from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .extractor import EntityRelationExtractor
from .graph import MarketGraph
from .models import Document, Entity, Relation


def load_documents(path: str | Path) -> list[Document]:
    documents = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                documents.append(Document(str(row["id"]), str(row["title"]), str(row["text"]), str(row["source"]), str(row["published_at"])))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid JSONL document on line {line_number}") from exc
    if not documents:
        raise ValueError("document collection is empty")
    if len({document.id for document in documents}) != len(documents):
        raise ValueError("document ids must be unique")
    return documents


class KnowledgeBase:
    def __init__(self, graph: MarketGraph | None = None, documents: list[Document] | None = None) -> None:
        self.graph = graph or MarketGraph()
        self.documents = documents or []

    @classmethod
    def build(cls, documents: list[Document], extractor: EntityRelationExtractor | None = None) -> "KnowledgeBase":
        graph = MarketGraph()
        extractor = extractor or EntityRelationExtractor()
        for document in documents:
            extractor.extract_into(graph, document)
        return cls(graph, documents)

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": 1,
            "documents": [asdict(item) for item in self.documents],
            "entities": [asdict(item) for item in self.graph.entities.values()],
            "relations": [asdict(item) for item in self.graph.relations],
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported knowledge base format")
        documents = [Document(**row) for row in payload["documents"]]
        entities = [Entity(**{**row, "aliases": tuple(row.get("aliases", []))}) for row in payload["entities"]]
        relations = [Relation(**row) for row in payload["relations"]]
        return cls(MarketGraph(entities, relations), documents)

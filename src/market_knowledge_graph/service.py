from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .models import QueryAnswer
from .retrieval import LocalGraphRAG
from .store import KnowledgeBase, load_documents


class SemanticProvider(Protocol):
    name: str

    def query(self, question: str) -> dict[str, Any]: ...


class KnowledgeGraphService:
    def __init__(self, base: KnowledgeBase, semantic_provider: SemanticProvider | None = None) -> None:
        self.base = base
        self.retriever = LocalGraphRAG(base)
        self.semantic_provider = semantic_provider

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "KnowledgeGraphService":
        return cls(KnowledgeBase.build(load_documents(path)))

    def ask(self, question: str, top_k: int = 4) -> QueryAnswer:
        local = self.retriever.query(question, top_k)
        if self.semantic_provider is None:
            return local
        semantic = self.semantic_provider.query(question)
        if not isinstance(semantic, dict) or not isinstance(semantic.get("answer"), str) or not semantic["answer"].strip():
            raise ValueError("semantic provider returned an invalid answer")
        sources = semantic.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("semantic provider returned no source metadata")
        source_ids = {str(source.get("id")) for source in sources if isinstance(source, dict) and source.get("id")}
        evidence_ids = {item.document_id for item in local.evidence}
        if not source_ids & evidence_ids:
            raise ValueError("semantic provider sources do not match the attached local evidence")
        return replace(
            local,
            answer=semantic["answer"].strip(),
            mode=str(semantic.get("mode") or getattr(self.semantic_provider, "name", "semantic-rag")),
            cautions=(*local.cautions, "Semantic wording was model-generated; verify it against the attached local evidence."),
        )

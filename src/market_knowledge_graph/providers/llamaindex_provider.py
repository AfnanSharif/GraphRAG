from __future__ import annotations

from typing import Any

from ..store import KnowledgeBase


class LlamaIndexRAG:
    """Optional semantic retrieval/generation over the same checked-in documents."""

    name = "llamaindex-openai"

    def __init__(self, base: KnowledgeBase, api_key: str, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small") -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required for LlamaIndex mode")
        try:
            from llama_index.core import Document, Settings, VectorStoreIndex
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.llms.openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the LlamaIndex/OpenAI optional dependencies") from exc
        Settings.llm = OpenAI(model=model, api_key=api_key, temperature=0)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=api_key)
        self.base = base
        source_by_id = {item.id: item for item in base.documents}
        documents = [Document(text=item.text, metadata={"id": item.id, "title": item.title, "source": item.source, "published_at": item.published_at, "kind": "source"}) for item in base.documents]
        for relation in base.graph.relations:
            source = base.graph.entities[relation.source].name
            target = base.graph.entities[relation.target].name
            origin = source_by_id.get(relation.document_id)
            documents.append(Document(
                text=f"Verified graph relation: {source} {relation.kind.replace('_', ' ')} {target}. Evidence: {relation.evidence}",
                metadata={
                    "id": relation.document_id,
                    "title": origin.title if origin else "Knowledge graph relation",
                    "source": origin.source if origin else "local-graph",
                    "published_at": origin.published_at if origin else "unknown",
                    "kind": "graph-relation",
                },
            ))
        self.engine: Any = VectorStoreIndex.from_documents(documents).as_query_engine(similarity_top_k=4)

    def query(self, question: str) -> dict[str, Any]:
        graph_context = self.graph_context(self.base, question)
        response = self.engine.query(
            "Answer only from retrieved source or verified graph-relation documents, cite source titles, distinguish fact from inference, "
            "and do not provide personalized financial advice.\n\n"
            + (f"Relevant graph neighborhood:\n{graph_context}\n\n" if graph_context else "")
            + question
        )
        sources = [{"score": node.score, **node.node.metadata} for node in response.source_nodes]
        answer = str(response).strip()
        if not answer or not sources:
            raise ValueError("LlamaIndex returned no grounded answer or sources")
        return {"answer": answer, "sources": sources, "mode": self.name}

    @staticmethod
    def graph_context(base: KnowledgeBase, question: str, max_facts: int = 12) -> str:
        facts: list[str] = []
        for entity in base.graph.find_entities(question):
            for relation, neighbor in base.graph.neighbors(entity.id):
                source = base.graph.entities[relation.source].name
                target = base.graph.entities[relation.target].name
                fact = f"{source} {relation.kind.replace('_', ' ')} {target} (document {relation.document_id})"
                if fact not in facts:
                    facts.append(fact)
                if len(facts) >= max_facts:
                    return "\n".join(f"- {item}" for item in facts)
        return "\n".join(f"- {item}" for item in facts)

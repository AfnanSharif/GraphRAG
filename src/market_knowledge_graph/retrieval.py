from __future__ import annotations

import math
import re
from collections import Counter

from .models import Document, Evidence, QueryAnswer
from .store import KnowledgeBase

STOP = {"about", "and", "are", "as", "at", "by", "did", "does", "for", "from", "how", "in", "into", "is", "it", "its", "most", "of", "on", "or", "the", "their", "this", "those", "to", "was", "were", "what", "when", "where", "which", "who", "with"}
POSITIVE = {"benefit", "beat", "growth", "higher", "improved", "profit", "record", "resilient", "rising", "strong", "surged", "upgraded"}
NEGATIVE = {"decline", "downgraded", "falling", "fell", "lower", "loss", "risk", "slowed", "weak", "warning"}


def tokenize(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]{1,}", text) if word.lower() not in STOP]


class LocalGraphRAG:
    def __init__(self, base: KnowledgeBase) -> None:
        self.base = base

    def query(self, question: str, top_k: int = 4) -> QueryAnswer:
        if not question.strip():
            raise ValueError("question is required")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be an integer between 1 and 50")
        ranked = self._rank(question)[:top_k]
        if not ranked or ranked[0][0] <= 0:
            raise ValueError("No grounded evidence matched the question. Try naming a company, sector, relationship, or market trend.")
        evidence = tuple(self._evidence(document, score, question) for score, document in ranked if score > 0)
        matched_entities = self.base.graph.find_entities(question)
        facts = self._facts(matched_entities, {item.document_id for item in evidence})
        if len(matched_entities) >= 2:
            path = self.base.graph.shortest_path(matched_entities[0].id, matched_entities[1].id)
            for relation in path:
                statement = self._relation_text(relation)
                if statement not in facts:
                    facts.append(statement)
        sentences = []
        for item in evidence[:3]:
            sentence = item.excerpt.rstrip(" .") + "."
            if sentence not in sentences:
                sentences.append(sentence)
        lead = "The indexed evidence indicates: " + " ".join(sentences)
        if facts:
            lead += " The graph also connects: " + "; ".join(facts[:4]) + "."
        score = self._sentiment(" ".join(item.excerpt for item in evidence))
        label = "positive" if score > .12 else "negative" if score < -.12 else "mixed/neutral"
        return QueryAnswer(question, lead, evidence, tuple(facts), label, score)

    def _rank(self, question: str) -> list[tuple[float, Document]]:
        query_terms = tokenize(question)
        documents = self.base.documents
        frequencies = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        document_frequency = Counter(term for term in set(query_terms) for counts in frequencies if term in counts)
        ranked = []
        for doc, counts in zip(documents, frequencies):
            length = sum(counts.values()) or 1
            score = 0.0
            for term in query_terms:
                if counts[term]:
                    inverse = math.log((len(documents) + 1) / (document_frequency[term] + .5)) + 1
                    score += inverse * counts[term] / math.sqrt(length)
            if any(entity.name.lower() in doc.text.lower() for entity in self.base.graph.find_entities(question)):
                score *= 1.5
            ranked.append((round(score, 6), doc))
        return sorted(ranked, key=lambda item: (item[0], item[1].published_at), reverse=True)

    @staticmethod
    def _evidence(document: Document, score: float, question: str) -> Evidence:
        terms = set(tokenize(question))
        options = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", document.text) if sentence.strip()]
        excerpt = max(options, key=lambda sentence: len(terms & set(tokenize(sentence))), default=document.text[:300])
        return Evidence(document.id, document.title, document.source, document.published_at, excerpt, score)

    def _facts(self, entities, document_ids: set[str]) -> list[str]:
        facts = []
        for entity in entities:
            for relation, _neighbor in self.base.graph.neighbors(entity.id):
                if relation.document_id in document_ids:
                    text = self._relation_text(relation)
                    if text not in facts:
                        facts.append(text)
        return facts

    def _relation_text(self, relation) -> str:
        source = self.base.graph.entities[relation.source].name
        target = self.base.graph.entities[relation.target].name
        return f"{source} {relation.kind.replace('_', ' ')} {target}"

    @staticmethod
    def _sentiment(text: str) -> float:
        terms = tokenize(text)
        if not terms:
            return 0.0
        raw = sum(term in POSITIVE for term in terms) - sum(term in NEGATIVE for term in terms)
        return round(max(-1.0, min(1.0, raw / max(3, math.sqrt(len(terms))))), 3)

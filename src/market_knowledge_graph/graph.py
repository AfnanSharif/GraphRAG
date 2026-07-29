from __future__ import annotations

import re
from collections import deque

from .models import Entity, Relation


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class MarketGraph:
    def __init__(self, entities: list[Entity] | None = None, relations: list[Relation] | None = None) -> None:
        self.entities: dict[str, Entity] = {entity.id: entity for entity in entities or []}
        self.relations: list[Relation] = list(relations or [])

    def add_entity(self, name: str, kind: str = "organization", aliases: tuple[str, ...] = ()) -> Entity:
        identifier = slug(name)
        existing = self.entities.get(identifier)
        if existing:
            merged = tuple(dict.fromkeys((*existing.aliases, *aliases)))
            entity = Entity(identifier, existing.name, existing.kind if existing.kind != "organization" else kind, merged)
        else:
            entity = Entity(identifier, name.strip(), kind, aliases)
        self.entities[identifier] = entity
        return entity

    def add_relation(self, source: Entity, target: Entity, kind: str, document_id: str, evidence: str) -> None:
        relation = Relation(source.id, target.id, kind, document_id, evidence.strip())
        if relation not in self.relations:
            self.relations.append(relation)

    def neighbors(self, entity_id: str) -> list[tuple[Relation, Entity]]:
        found = []
        for relation in self.relations:
            if relation.source == entity_id and relation.target in self.entities:
                found.append((relation, self.entities[relation.target]))
            elif relation.target == entity_id and relation.source in self.entities:
                found.append((relation, self.entities[relation.source]))
        return found

    def find_entities(self, text: str) -> list[Entity]:
        lowered = text.lower()
        matches = []
        for entity in self.entities.values():
            names = (entity.name, *entity.aliases)
            if any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in names):
                matches.append(entity)
        return sorted(matches, key=lambda item: len(item.name), reverse=True)

    def shortest_path(self, source_id: str, target_id: str, max_hops: int = 4) -> list[Relation]:
        queue = deque([(source_id, [])])
        seen = {source_id}
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for relation, neighbor in self.neighbors(current):
                next_path = [*path, relation]
                if neighbor.id == target_id:
                    return next_path
                if neighbor.id not in seen:
                    seen.add(neighbor.id)
                    queue.append((neighbor.id, next_path))
        return []

    def to_dot(self, focus: set[str] | None = None) -> str:
        focus = focus or set(self.entities)
        related = set(focus)
        for relation in self.relations:
            if relation.source in focus or relation.target in focus:
                related.update({relation.source, relation.target})
        colors = {"company": "#67e8f9", "sector": "#fbbf24", "trend": "#c084fc", "organization": "#94a3b8"}
        lines = ['digraph G {', 'graph [bgcolor="transparent", rankdir="LR", pad="0.3"];', 'node [shape="box", style="rounded,filled", fontname="Arial", fontcolor="#081018"];', 'edge [fontname="Arial", color="#64748b", fontcolor="#94a3b8"];']
        for identifier in sorted(related):
            entity = self.entities[identifier]
            label = entity.name.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            lines.append(f'"{identifier}" [label="{label}", fillcolor="{colors.get(entity.kind, colors["organization"])}"];')
        for relation in self.relations:
            if relation.source in related and relation.target in related:
                label = relation.kind.replace("_", " ").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
                lines.append(f'"{relation.source}" -> "{relation.target}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)

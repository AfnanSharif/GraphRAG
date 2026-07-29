from __future__ import annotations

from ..store import KnowledgeBase


class Neo4jPublisher:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install neo4j to publish the graph") from exc
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def publish(self, base: KnowledgeBase) -> dict[str, int]:
        with self.driver.session(database=self.database) as session:
            for entity in base.graph.entities.values():
                session.run("MERGE (n:Entity {id:$id}) SET n.name=$name, n.kind=$kind, n.aliases=$aliases", id=entity.id, name=entity.name, kind=entity.kind, aliases=list(entity.aliases))
            for relation in base.graph.relations:
                session.run(
                    "MATCH (a:Entity {id:$source}), (b:Entity {id:$target}) MERGE (a)-[r:RELATED {kind:$kind, document_id:$document_id}]->(b) SET r.evidence=$evidence",
                    source=relation.source, target=relation.target, kind=relation.kind, document_id=relation.document_id, evidence=relation.evidence,
                )
        return {"entities": len(base.graph.entities), "relations": len(base.graph.relations)}

    def close(self) -> None:
        self.driver.close()

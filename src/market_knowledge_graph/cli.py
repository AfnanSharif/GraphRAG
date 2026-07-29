from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .providers import (
    create_market_collector,
    create_neo4j_publisher,
    create_rss_collector,
    create_search_collector,
    create_semantic_provider,
    create_web_collector,
)
from .service import KnowledgeGraphService
from .store import KnowledgeBase, load_documents


def _write_documents(documents, output: Path) -> dict[str, object]:
    unique = {document.id: document for document in documents}
    if not unique:
        raise ValueError("collector returned no usable documents")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(asdict(document), ensure_ascii=False) + "\n" for document in unique.values()), encoding="utf-8")
    return {"output": str(output), "documents": len(unique)}


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Build and query a financial knowledge graph")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("documents", type=Path)
    build.add_argument("--output", type=Path, default=Path(os.getenv("KNOWLEDGE_BASE_PATH", ".local/knowledge-base.json")))
    query = commands.add_parser("query")
    query.add_argument("knowledge_base", type=Path)
    query.add_argument("question")
    query.add_argument("--provider", choices=["local", "llamaindex"], default=os.getenv("RAG_PROVIDER", "local"))
    query.add_argument("--top-k", type=int, default=4)
    stats = commands.add_parser("stats")
    stats.add_argument("knowledge_base", type=Path)
    publish = commands.add_parser("publish-neo4j", help="publish a persisted graph to configured Neo4j")
    publish.add_argument("knowledge_base", type=Path)
    collect = commands.add_parser("collect-rss", help="collect bounded RSS excerpts into JSONL")
    collect.add_argument("urls", nargs="+")
    collect.add_argument("--output", type=Path, default=Path(".local/rss-documents.jsonl"))
    market = commands.add_parser("collect-market", help="collect bounded Alpha Vantage quote documents")
    market.add_argument("symbols", nargs="+")
    market.add_argument("--output", type=Path, default=Path(".local/market-documents.jsonl"))
    search = commands.add_parser("collect-search", help="collect Google Programmable Search snippets")
    search.add_argument("query")
    search.add_argument("--output", type=Path, default=Path(".local/search-documents.jsonl"))
    web = commands.add_parser("collect-web", help="extract bounded text from explicit public article URLs")
    web.add_argument("urls", nargs="+")
    web.add_argument("--output", type=Path, default=Path(".local/web-documents.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "build":
        base = KnowledgeBase.build(load_documents(args.documents))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        base.save(args.output)
        print(json.dumps({"output": str(args.output), "documents": len(base.documents), "entities": len(base.graph.entities), "relations": len(base.graph.relations)}, indent=2))
    elif args.command in {"stats", "query", "publish-neo4j"}:
        base = KnowledgeBase.load(args.knowledge_base)
        if args.command == "stats":
            print(json.dumps({"documents": len(base.documents), "entities": len(base.graph.entities), "relations": len(base.graph.relations)}, indent=2))
        elif args.command == "query":
            semantic = create_semantic_provider(args.provider, base)
            print(json.dumps(KnowledgeGraphService(base, semantic).ask(args.question, args.top_k).to_dict(), indent=2, ensure_ascii=False))
        else:
            publisher = create_neo4j_publisher()
            try:
                print(json.dumps(publisher.publish(base), indent=2))
            finally:
                publisher.close()
    elif args.command == "collect-rss":
        collector = create_rss_collector()
        documents = [document for url in args.urls for document in collector.collect(url)]
        print(json.dumps(_write_documents(documents, args.output), indent=2))
    elif args.command == "collect-market":
        print(json.dumps(_write_documents(create_market_collector().collect(args.symbols), args.output), indent=2))
    elif args.command == "collect-search":
        print(json.dumps(_write_documents(create_search_collector().collect(args.query), args.output), indent=2))
    else:
        print(json.dumps(_write_documents(create_web_collector().collect(args.urls), args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

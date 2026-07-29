import os

from .llamaindex_provider import LlamaIndexRAG
from .neo4j_provider import Neo4jPublisher
from .rss import RSSCollector
from .market_sources import AlphaVantageCollector, GoogleSearchCollector, WebArticleCollector

__all__ = ["AlphaVantageCollector", "GoogleSearchCollector", "LlamaIndexRAG", "Neo4jPublisher", "RSSCollector", "WebArticleCollector"]


def create_semantic_provider(name: str | None, base):
    selected = (name or os.getenv("RAG_PROVIDER", "local")).strip().lower()
    if selected in {"", "local", "none"}:
        return None
    if selected == "llamaindex":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LlamaIndex mode")
        return LlamaIndexRAG(
            base,
            api_key,
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    raise ValueError("RAG_PROVIDER must be local or llamaindex")


def create_neo4j_publisher():
    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "")
    if not uri or not username or not password:
        raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are required")
    return Neo4jPublisher(uri, username, password, os.getenv("NEO4J_DATABASE", "neo4j"))


def create_rss_collector():
    return RSSCollector(
        timeout=float(os.getenv("RSS_TIMEOUT_SECONDS", "10")),
        max_items=int(os.getenv("RSS_MAX_ITEMS", "25")),
        user_agent=os.getenv("RSS_USER_AGENT", "MarketGraphResearchBot/1.0"),
    )


def create_market_collector():
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is required")
    return AlphaVantageCollector(key, float(os.getenv("SOURCE_TIMEOUT_SECONDS", "10")))


def create_search_collector():
    key, engine = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip(), os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
    if not key or not engine:
        raise ValueError("GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID are required")
    return GoogleSearchCollector(key, engine, float(os.getenv("SOURCE_TIMEOUT_SECONDS", "10")), int(os.getenv("GOOGLE_SEARCH_MAX_ITEMS", "10")))


def create_web_collector():
    return WebArticleCollector(float(os.getenv("SOURCE_TIMEOUT_SECONDS", "10")), int(os.getenv("WEB_MAX_CHARS", "12000")))


__all__ += ["create_market_collector", "create_search_collector", "create_semantic_provider", "create_neo4j_publisher", "create_rss_collector", "create_web_collector"]

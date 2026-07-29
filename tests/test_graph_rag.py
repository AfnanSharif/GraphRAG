import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_knowledge_graph.cli import main as cli_main
from market_knowledge_graph.models import Document
from market_knowledge_graph.providers.llamaindex_provider import LlamaIndexRAG
from market_knowledge_graph.providers.market_sources import AlphaVantageCollector, GoogleSearchCollector, WebArticleCollector
from market_knowledge_graph.service import KnowledgeGraphService
from market_knowledge_graph.store import KnowledgeBase, load_documents


class GraphRAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = Path(__file__).resolve().parents[1] / "data" / "market_news.jsonl"
        cls.documents = load_documents(cls.sample)
        cls.base = KnowledgeBase.build(cls.documents)

    def test_extraction_builds_expected_relationships(self):
        self.assertIn("aster-cloud", self.base.graph.entities)
        kinds = {relation.kind for relation in self.base.graph.relations}
        self.assertIn("partners_with", kinds)
        self.assertIn("supplies", kinds)
        self.assertIn("operates_in", kinds)

    def test_path_between_companies(self):
        path = self.base.graph.shortest_path("northstar-bank", "aster-cloud")
        self.assertTrue(path)
        self.assertLessEqual(len(path), 2)

    def test_grounded_query_has_evidence_and_facts(self):
        answer = KnowledgeGraphService(self.base).ask("How is Aster Cloud connected to Northstar Bank?")
        self.assertTrue(answer.evidence)
        self.assertTrue(any("partners with" in fact for fact in answer.graph_facts))
        self.assertTrue(all(item.document_id.startswith("brief-") for item in answer.evidence))
        json.dumps(answer.to_dict())

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.json"
            self.base.save(path)
            restored = KnowledgeBase.load(path)
            self.assertEqual(len(restored.graph.entities), len(self.base.graph.entities))
            self.assertEqual(len(restored.graph.relations), len(self.base.graph.relations))

    def test_unknown_topic_is_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeGraphService(self.base).ask("volcanic geology in antarctica")

    def test_invalid_retrieval_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeGraphService(self.base).ask("Aster Cloud", top_k=-1)

    def test_semantic_provider_is_invoked_and_local_evidence_is_preserved(self):
        class FakeSemanticProvider:
            name = "fake-semantic"

            def __init__(self):
                self.questions = []

            def query(self, question):
                self.questions.append(question)
                return {"answer": "Provider synthesis", "sources": [{"id": "brief-001"}], "mode": self.name}

        provider = FakeSemanticProvider()
        answer = KnowledgeGraphService(self.base, provider).ask("How is Aster Cloud connected to Northstar Bank?")
        self.assertEqual(provider.questions, ["How is Aster Cloud connected to Northstar Bank?"])
        self.assertEqual(answer.answer, "Provider synthesis")
        self.assertEqual(answer.mode, "fake-semantic")
        self.assertTrue(answer.evidence)
        self.assertTrue(answer.graph_facts)

    def test_llamaindex_prompt_context_contains_graph_neighbors_and_provenance(self):
        context = LlamaIndexRAG.graph_context(self.base, "Aster Cloud and Northstar Bank")
        self.assertIn("Aster Cloud", context)
        self.assertIn("Northstar Bank", context)
        self.assertIn("document brief-", context)

    def test_market_and_search_collectors_convert_live_contracts_to_documents(self):
        market = AlphaVantageCollector(
            "test-key",
            fetcher=lambda symbol: {
                "Global Quote": {
                    "05. price": "174.25",
                    "10. change percent": "+1.2%",
                    "06. volume": "12000",
                    "07. latest trading day": "2026-07-28",
                }
            },
        ).collect(["ACME"])
        search = GoogleSearchCollector(
            "test-key",
            "test-engine",
            fetcher=lambda query: {
                "items": [{"title": "Market report", "snippet": "ACME expanded its cloud partnership.", "link": "https://example.test/report"}]
            },
        ).collect("ACME cloud")
        self.assertEqual(market[0].id, "quote-acme-2026-07-28")
        self.assertIn("174.25", market[0].text)
        self.assertEqual(search[0].source, "https://example.test/report")

    def test_web_collector_enforces_public_urls_and_bounds_extracted_text(self):
        collector = WebArticleCollector(max_chars=500, fetcher=lambda url: ("Research", "word " * 200))
        documents = collector.collect(["https://example.test/article"])
        self.assertEqual(documents[0].title, "Research")
        self.assertLessEqual(len(documents[0].text), 500)
        with self.assertRaises(ValueError):
            collector.collect(["http://127.0.0.1/admin"])

    def test_publish_subcommand_invokes_neo4j_adapter_and_closes_it(self):
        class FakePublisher:
            def __init__(self):
                self.published = 0
                self.closed = False

            def publish(self, base):
                self.published += 1
                return {"entities": len(base.graph.entities), "relations": len(base.graph.relations)}

            def close(self):
                self.closed = True

        publisher = FakePublisher()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.json"
            self.base.save(path)
            with patch("market_knowledge_graph.cli.create_neo4j_publisher", return_value=publisher), redirect_stdout(StringIO()):
                self.assertEqual(cli_main(["publish-neo4j", str(path)]), 0)
        self.assertEqual(publisher.published, 1)
        self.assertTrue(publisher.closed)

    def test_collect_subcommand_invokes_rss_adapter_and_writes_jsonl(self):
        class FakeCollector:
            def __init__(self):
                self.urls = []

            def collect(self, url):
                self.urls.append(url)
                return [Document("feed-1", "Feed item", "Licensed excerpt", url, "2026-07-28")]

        collector = FakeCollector()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "feed.jsonl"
            with patch("market_knowledge_graph.cli.create_rss_collector", return_value=collector), redirect_stdout(StringIO()):
                self.assertEqual(cli_main(["collect-rss", "https://example.test/feed.xml", "--output", str(output)]), 0)
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(collector.urls, ["https://example.test/feed.xml"])
        self.assertEqual(row["id"], "feed-1")

    def test_market_collection_is_reachable_from_cli(self):
        class FakeMarketCollector:
            def collect(self, symbols):
                self.symbols = list(symbols)
                return [Document("quote-acme", "ACME quote", "ACME traded at 42.", "https://example.test/quote", "2026-07-28")]

        collector = FakeMarketCollector()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "market.jsonl"
            with patch("market_knowledge_graph.cli.create_market_collector", return_value=collector), redirect_stdout(StringIO()):
                self.assertEqual(cli_main(["collect-market", "ACME", "BETA", "--output", str(output)]), 0)
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(collector.symbols, ["ACME", "BETA"])
        self.assertEqual(row["id"], "quote-acme")


if __name__ == "__main__":
    unittest.main()

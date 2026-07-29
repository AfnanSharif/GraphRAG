from __future__ import annotations

import json
import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from market_knowledge_graph.providers import create_semantic_provider
from market_knowledge_graph.service import KnowledgeGraphService
from market_knowledge_graph.store import KnowledgeBase, load_documents


@st.cache_resource(show_spinner=False)
def _query_service(provider_name: str, document_ids: tuple[str, ...], _base: KnowledgeBase) -> KnowledgeGraphService:
    return KnowledgeGraphService(_base, create_semantic_provider(provider_name, _base))

st.set_page_config(page_title="MarketMesh · Graph RAG", page_icon="⬡", layout="wide")
st.markdown("""<style>
@keyframes breathe{50%{box-shadow:0 0 80px #c084fc22}}.stApp{background:radial-gradient(circle at 70% 0,#25306455,transparent 32%),#080b14;color:#eef2ff}.hero{padding:2.5rem;border:1px solid #ffffff1a;border-radius:26px;background:#101525cc;animation:breathe 5s ease-in-out infinite}.brand{color:#c084fc;letter-spacing:.18em;font-weight:900}.evidence{padding:1rem;border-radius:14px;background:#101623;border:1px solid #ffffff14;margin:.7rem 0}.source{color:#67e8f9;font:700 .75rem monospace}.fact{display:inline-block;padding:.35rem .7rem;margin:.2rem;border:1px solid #c084fc55;border-radius:99px;color:#ddd6fe}
[data-testid="stSidebar"]{background:#0c101b}div.stButton>button{background:#c084fc;color:#0a0710;border:0;border-radius:99px;font-weight:800}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style><div class="hero"><div class="brand">MARKETMESH</div><h1>Research the market through relationships—not isolated headlines.</h1><p>Grounded local retrieval, inspectable graph facts, provenance, and optional LlamaIndex/OpenAI depth.</p></div>""", unsafe_allow_html=True)

sample = ROOT / "data" / "market_news.jsonl"
index_path = Path(os.getenv("KNOWLEDGE_BASE_PATH", ROOT / ".local" / "knowledge-base.json"))
if index_path.exists():
    base = KnowledgeBase.load(index_path)
else:
    base = KnowledgeBase.build(load_documents(sample))
with st.sidebar:
    st.header("Knowledge base")
    st.metric("Documents", len(base.documents))
    st.metric("Entities", len(base.graph.entities))
    st.metric("Relations", len(base.graph.relations))
    entity_name = st.selectbox("Explore entity", ["All"] + sorted(entity.name for entity in base.graph.entities.values()))
    provider_options = ["local", "llamaindex"]
    configured_provider = os.getenv("RAG_PROVIDER", "local").lower()
    provider_name = st.selectbox("Answer provider", provider_options, index=provider_options.index(configured_provider) if configured_provider in provider_options else 0)
    st.caption("The bundled sample is fictional and dated. Build an index from your own licensed sources for real analysis.")

st.markdown("### Ask a relationship-aware question")
ideas = ["How is Aster Cloud connected to Northstar Bank?", "What risks affect Helio Grid?", "What trends are reported for cloud infrastructure?"]
columns = st.columns(3)
for column, idea in zip(columns, ideas):
    if column.button(idea, use_container_width=True):
        st.session_state.question = idea
question = st.text_input("Question", st.session_state.get("question", ""), placeholder="Ask about companies, sectors, partners, trends, or risk…", label_visibility="collapsed")
if st.button("Trace the evidence", type="primary"):
    try:
        service = _query_service(provider_name, tuple(document.id for document in base.documents), base)
        st.session_state.answer = service.ask(question)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

left, right = st.columns([1.05, .95])
with left:
    st.markdown("### Relationship map")
    focus = None
    if entity_name != "All":
        focus = {next(entity.id for entity in base.graph.entities.values() if entity.name == entity_name)}
    st.graphviz_chart(base.graph.to_dot(focus), use_container_width=True)
with right:
    answer = st.session_state.get("answer")
    if answer:
        st.markdown("### Grounded synthesis")
        st.write(answer.answer)
        st.caption(f"Evidence tone: {answer.sentiment} ({answer.sentiment_score:+.3f}) · {answer.mode}")
        if answer.graph_facts:
            st.markdown("**Graph facts**")
            st.markdown("".join(f"<span class='fact'>{html.escape(fact)}</span>" for fact in answer.graph_facts), unsafe_allow_html=True)
        st.markdown("**Evidence trail**")
        for item in answer.evidence:
            st.markdown(f"<div class='evidence'><div class='source'>{html.escape(item.published_at)} · {html.escape(item.source)} · relevance {item.score:.3f}</div><b>{html.escape(item.title)}</b><p>{html.escape(item.excerpt)}</p></div>", unsafe_allow_html=True)
        for caution in answer.cautions:
            st.warning(caution)
        st.download_button("Export research JSON", json.dumps(answer.to_dict(), indent=2), "marketmesh-answer.json", "application/json")
    else:
        st.info("Choose an example or ask a question to see evidence and graph paths together.")

"""
GraphRAG Streamlit UI — Full Pipeline Debugging Dashboard

Pages:
1. Upload & Ingest — upload docs, see chunks/entities/relations/timing/tokens
2. Knowledge Graph — interactive graph visualization
3. Chat — ask questions with full retrieval debug panel
4. Pipeline Inspector — browse all chunks, entities, DB stats
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

# =============================================================================
# Configuration
# =============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
API = f"{BACKEND_URL}/api"

st.set_page_config(
    page_title="GraphRAG Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS — Dark theme, gradient cards, modern styling
# =============================================================================
st.markdown("""
<style>
/* --- Global --- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
/* --- Metrics cards --- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    background: linear-gradient(135deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* --- Expanders --- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
}
/* --- Custom pipeline step cards --- */
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #3a7bd5;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
}
.pipeline-step-done {
    border-left-color: #00c853;
}
.step-label {
    color: #888;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.step-value {
    color: #fff;
    font-size: 1.1rem;
    font-weight: 600;
}
/* --- Graph fact badges --- */
.graph-fact {
    background: linear-gradient(135deg, rgba(78,205,196,0.15), rgba(69,183,209,0.15));
    border: 1px solid rgba(78,205,196,0.3);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.9rem;
}
/* --- Timing bar highlight --- */
.timing-item {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

def api_get(endpoint: str):
    """GET request to backend API."""
    try:
        r = requests.get(f"{API}/{endpoint}", timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(endpoint: str, **kwargs):
    """POST request to backend API."""
    try:
        r = requests.post(f"{API}/{endpoint}", timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Is it running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_delete(endpoint: str):
    """DELETE request to backend API."""
    try:
        r = requests.delete(f"{API}/{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# Entity type color map (consistent across all pages)
TYPE_COLORS = {
    "Person": "#FF6B6B",
    "Organization": "#4ECDC4",
    "Concept": "#45B7D1",
    "Location": "#96CEB4",
    "Event": "#FFEAA7",
    "Policy": "#DDA0DD",
    "Technology": "#98D8C8",
    "Other": "#C9C9C9",
}


def render_legend():
    """Render entity type color legend."""
    cols = st.columns(len(TYPE_COLORS))
    for i, (etype, color) in enumerate(TYPE_COLORS.items()):
        cols[i].markdown(
            f'<span style="color:{color}; font-size:18px">●</span> {etype}',
            unsafe_allow_html=True,
        )


# =============================================================================
# Sidebar — Navigation + System Status
# =============================================================================

st.sidebar.title("🔬 GraphRAG")
st.sidebar.caption("Graph-Enhanced RAG Demo")

page = st.sidebar.radio(
    "Navigate",
    ["📤 Upload & Ingest", "🕸️ Knowledge Graph", "💬 Chat", "🔍 Pipeline Inspector"],
)

# System health
st.sidebar.divider()
st.sidebar.subheader("System Status")
health = api_get("health")
if health:
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Database", "✅" if health["db"] == "connected" else "❌")
    col2.metric("LLM", "✅" if health["llm"] == "connected" else "❌")
    with st.sidebar.expander("⚙️ Config"):
        st.json(health.get("config", {}))

# Reset button in sidebar
st.sidebar.divider()
if st.sidebar.button("🗑️ Reset All Data", type="secondary", use_container_width=True):
    result = api_delete("reset")
    if result:
        st.sidebar.success("All data cleared!")
        st.rerun()


# =============================================================================
# Page 1: Upload & Ingest
# =============================================================================

if page == "📤 Upload & Ingest":
    st.title("📤 Document Upload & Ingestion")
    st.markdown("Upload a **PDF**, **DOCX**, or **TXT** file to ingest into the GraphRAG system.")

    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    if uploaded and st.button("🚀 Ingest Document", type="primary"):
        progress = st.progress(0, text="Starting ingestion pipeline...")

        with st.spinner("Processing..."):
            start = time.time()
            progress.progress(10, text="📄 Uploading to backend...")
            result = api_post("ingest", files={"file": (uploaded.name, uploaded.read(), uploaded.type)})
            total_time = round(time.time() - start, 2)
            progress.progress(100, text="✅ Ingestion complete!")

        if result:
            st.success(f"✅ Ingested **{result['filename']}** — {result['total_chunks']} chunks, "
                       f"{len(result['entities'])} entities, {len(result['relations'])} relations "
                       f"in {total_time}s")

            # --- Pipeline Overview ---
            st.subheader("📊 Pipeline Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📦 Chunks", result['total_chunks'])
            col2.metric("🏷️ Entities", len(result['entities']))
            col3.metric("🔗 Relations", len(result['relations']))
            tok = result.get("token_usage", {})
            col4.metric("🪙 Tokens Used", tok.get("total", 0))

            # --- Timing Breakdown ---
            st.subheader("⏱️ Pipeline Timing")
            timing = result.get("timing", {})
            timing_df = pd.DataFrame([
                {"Step": k.replace("_s", "").replace("_", " ").title(), "Seconds": v}
                for k, v in timing.items()
            ])
            st.bar_chart(timing_df.set_index("Step"))

            # --- Token Usage Detail ---
            st.subheader("🪙 Token Usage (Entity Extraction)")
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("Prompt Tokens", tok.get("prompt_tokens", 0))
            tcol2.metric("Completion Tokens", tok.get("completion_tokens", 0))
            tcol3.metric("Total Tokens", tok.get("total", 0))

            # --- Chunks ---
            st.subheader(f"📦 Chunks ({result['total_chunks']})")
            for chunk in result["chunks"]:
                with st.expander(f"Chunk {chunk['chunk_index']} — {len(chunk['content'])} chars"):
                    st.text(chunk["content"])
                    st.caption(f"Embedding preview: {chunk['embedding_preview']}")

            # --- Entities ---
            st.subheader(f"🏷️ Entities ({len(result['entities'])})")
            if result["entities"]:
                ent_df = pd.DataFrame(result["entities"])
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

            # --- Relations (as a graph-like table) ---
            st.subheader(f"🔗 Relations ({len(result['relations'])})")
            if result["relations"]:
                for rel in result["relations"]:
                    st.markdown(
                        f'<div class="graph-fact">'
                        f'<strong>{rel["source"]}</strong> '
                        f'→ <code>{rel["relation"]}</code> → '
                        f'<strong>{rel["target"]}</strong> '
                        f'<span style="color:#888">(chunk {rel["source_chunk"]})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Show existing documents
    st.divider()
    st.subheader("📚 Ingested Documents")
    docs = api_get("documents")
    if docs and docs.get("documents"):
        for doc in docs["documents"]:
            st.write(f"**{doc['filename']}** — {doc['chunk_count']} chunks "
                     f"(ingested {doc['created_at']})")
    else:
        st.info("No documents ingested yet. Upload one above!")


# =============================================================================
# Page 2: Knowledge Graph
# =============================================================================

elif page == "🕸️ Knowledge Graph":
    st.title("🕸️ Knowledge Graph Visualization")

    graph_data = api_get("graph")
    if graph_data:
        nodes_data = graph_data.get("nodes", [])
        edges_data = graph_data.get("edges", [])

        if not nodes_data:
            st.info("No entities in the knowledge graph yet. Ingest a document first!")
        else:
            # Stats
            col1, col2 = st.columns(2)
            col1.metric("🏷️ Entities", len(nodes_data))
            col2.metric("🔗 Relations", len(edges_data))

            # Entity type filter
            all_types = sorted(set(n.get("type", "Other") for n in nodes_data))
            selected_types = st.multiselect(
                "Filter by entity type",
                all_types,
                default=all_types,
            )
            filtered_nodes = [n for n in nodes_data if n.get("type", "Other") in selected_types]
            filtered_names = set(n.get("name") for n in filtered_nodes)
            filtered_edges = [
                e for e in edges_data
                if e["source"] in filtered_names and e["target"] in filtered_names
            ]

            # Build agraph
            ag_nodes = []
            node_names = set()
            for n in filtered_nodes:
                name = n.get("name", "?")
                ntype = n.get("type", "Other")
                if name not in node_names:
                    node_names.add(name)
                    ag_nodes.append(Node(
                        id=name,
                        label=name,
                        size=25,
                        color=TYPE_COLORS.get(ntype, "#C9C9C9"),
                        title=f"{name} ({ntype})",
                    ))

            ag_edges = [
                Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e["relation"],
                    color="#888888",
                )
                for e in filtered_edges
            ]

            tab_interactive, tab_svg = st.tabs(["🕸️ Interactive Graph", "🖼️ Graphviz Layout"])

            with tab_interactive:
                config = Config(
                    width=900,
                    height=600,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                )
                agraph(nodes=ag_nodes, edges=ag_edges, config=config)

            with tab_svg:
                dot = "digraph {\n"
                dot += '  graph [rankdir=LR, bgcolor="transparent", margin=0];\n'
                dot += '  node [style=filled, fontname="Helvetica", shape=box, fontsize=11];\n'
                dot += '  edge [fontname="Helvetica", fontsize=9, color="#888888"];\n'
                for n in filtered_nodes:
                    name = n.get("name", "?")
                    ntype = n.get("type", "Other")
                    color = TYPE_COLORS.get(ntype, "#C9C9C9")
                    safe_name = name.replace('"', '\\"')
                    dot += f'  "{safe_name}" [fillcolor="{color}", label="{safe_name}\\n({ntype})"];\n'
                for e in filtered_edges:
                    src = e["source"].replace('"', '\\"')
                    tgt = e["target"].replace('"', '\\"')
                    rel = e["relation"].replace('"', '\\"')
                    dot += f'  "{src}" -> "{tgt}" [label="{rel}"];\n'
                dot += "}"
                st.graphviz_chart(dot, use_container_width=True)

            # Legend
            st.divider()
            st.subheader("🎨 Entity Types")
            render_legend()

            # Raw data
            with st.expander("📊 Raw Graph Data"):
                tab1, tab2 = st.tabs(["Nodes", "Edges"])
                with tab1:
                    st.dataframe(pd.DataFrame(filtered_nodes), use_container_width=True, hide_index=True)
                with tab2:
                    if filtered_edges:
                        st.dataframe(pd.DataFrame(filtered_edges), use_container_width=True, hide_index=True)
                    else:
                        st.info("No edges to display")


# =============================================================================
# Page 3: Chat
# =============================================================================

elif page == "💬 Chat":
    st.title("💬 Chat with Your Documents")

    # User selector
    user_id = st.selectbox("👤 User", ["user_a", "user_b"], help="Two-user demo — separate histories")

    # Chat history
    history = api_get(f"chat_history/{user_id}")
    if history and history.get("history"):
        for msg in history["history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("🔍 Running GraphRAG pipeline..."):
            result = api_post("chat", json={"question": question, "user_id": user_id})

        if result:
            # Show answer
            with st.chat_message("assistant"):
                st.write(result["answer"])

            # ============================================================
            # DEBUG PANEL — Full GraphRAG Pipeline Visibility
            # ============================================================
            debug = result.get("debug", {})

            st.divider()
            st.subheader("🔬 GraphRAG Pipeline Debug")

            # --- Pipeline Steps Overview ---
            timing = debug.get("timing", {})
            steps = [
                ("1️⃣ Embed Query", "embed_query_s", "Converted question to vector"),
                ("2️⃣ Vector Search", "vector_search_s", f"{len(debug.get('vector_results', []))} chunks found"),
                ("3️⃣ Graph Search", "graph_search_s", f"{len(debug.get('graph_facts', []))} facts found"),
                ("4️⃣ Answer Generation", "generation_s", "LLM generated answer"),
            ]

            cols = st.columns(4)
            for i, (label, key, desc) in enumerate(steps):
                t = timing.get(key, 0)
                cols[i].markdown(
                    f'<div class="pipeline-step pipeline-step-done">'
                    f'<div class="step-label">{label}</div>'
                    f'<div class="step-value">{t}s</div>'
                    f'<div style="color:#aaa;font-size:0.8rem">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # --- Token Usage ---
            tok = debug.get("token_usage", {})
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("Prompt Tokens", tok.get("prompt_tokens", 0))
            tcol2.metric("Completion Tokens", tok.get("completion_tokens", 0))
            tcol3.metric("Total Tokens", tok.get("total", 0))

            # --- Question Entities ---
            q_entities = debug.get("question_entities", [])
            if q_entities:
                st.subheader("🏷️ Extracted Question Entities")
                st.markdown(" ".join(
                    f'`{e}`' for e in q_entities
                ))

            # --- Graph Facts ---
            st.subheader(f"🕸️ Knowledge Graph Facts ({len(debug.get('graph_facts', []))})")
            facts = debug.get("graph_facts", [])
            if facts:
                for f in facts:
                    st.markdown(
                        f'<div class="graph-fact">'
                        f'<strong>{f["source"]}</strong> '
                        f'→ <code>{f["relation"]}</code> → '
                        f'<strong>{f["target"]}</strong>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No relevant graph facts found for this query")

            # --- Vector Search Results ---
            with st.expander(f"🔷 Vector Search Results ({len(debug.get('vector_results', []))})"):
                vec_res = debug.get("vector_results", [])
                if vec_res:
                    for i, r in enumerate(vec_res):
                        boost = " 🔗 **Graph-boosted**" if r.get("graph_boosted") else ""
                        st.markdown(f"**#{i+1}** — Cosine: `{r.get('vector_score', 'N/A')}` "
                                    f"| File: `{r.get('doc_filename', '')}`{boost}")
                        st.text(r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"])
                        st.divider()
                else:
                    st.info("No vector results")

            # --- Final Sources Used ---
            with st.expander(f"📄 Final Sources Used ({len(result.get('sources', []))})"):
                for i, s in enumerate(result.get("sources", [])):
                    boost = " 🔗 Graph-boosted" if s.get("graph_boosted") else ""
                    st.markdown(f"**Source {i+1}** (Chunk {s['chunk_id']}) "
                                f"from `{s['doc_filename']}`{boost}")
                    st.text(s["content"])
                    st.divider()


# =============================================================================
# Page 4: Pipeline Inspector
# =============================================================================

elif page == "🔍 Pipeline Inspector":
    st.title("🔍 Pipeline Inspector")
    st.markdown("Browse all data in the system — chunks, entities, relations, stats.")

    # Stats
    docs = api_get("documents")
    graph_data = api_get("graph")

    if docs and graph_data:
        col1, col2, col3, col4 = st.columns(4)
        doc_count = len(docs.get("documents", []))
        chunk_count = sum(d.get("chunk_count", 0) for d in docs.get("documents", []))
        node_count = len(graph_data.get("nodes", []))
        edge_count = len(graph_data.get("edges", []))

        col1.metric("📄 Documents", doc_count)
        col2.metric("📦 Chunks", chunk_count)
        col3.metric("🏷️ Entities", node_count)
        col4.metric("🔗 Relations", edge_count)

    # Browse chunks by document
    st.divider()
    st.subheader("📦 Chunks by Document")

    if docs and docs.get("documents"):
        doc_names = {d["id"]: d["filename"] for d in docs["documents"]}
        selected_doc = st.selectbox(
            "Select document",
            options=list(doc_names.keys()),
            format_func=lambda x: doc_names[x],
        )

        if selected_doc:
            chunks = api_get(f"chunks/{selected_doc}")
            if chunks and chunks.get("chunks"):
                for c in chunks["chunks"]:
                    with st.expander(f"Chunk {c['chunk_index']} — {len(c['content'])} chars"):
                        st.text(c["content"])
    else:
        st.info("No documents ingested yet.")

    # All entities
    st.divider()
    st.subheader("🏷️ All Entities")
    if graph_data and graph_data.get("nodes"):
        st.dataframe(pd.DataFrame(graph_data["nodes"]), use_container_width=True, hide_index=True)
    else:
        st.info("No entities yet")

    # All relations
    st.subheader("🔗 All Relations")
    if graph_data and graph_data.get("edges"):
        st.dataframe(pd.DataFrame(graph_data["edges"]), use_container_width=True, hide_index=True)
    else:
        st.info("No relations yet")

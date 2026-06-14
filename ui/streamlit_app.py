"""
GraphRAG Streamlit UI — Full Pipeline Debugging Dashboard

Pages:
1. Upload & Ingest — upload docs, see chunks/entities/relations/timing/tokens
2. Knowledge Graph — interactive graph visualization
3. Chat — ask questions with full retrieval debug panel
4. Pipeline Inspector — browse all chunks, entities, DB stats
"""

import os
import json
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
    with st.sidebar.expander("Config"):
        st.json(health.get("config", {}))


# =============================================================================
# Page 1: Upload & Ingest
# =============================================================================

if page == "📤 Upload & Ingest":
    st.title("📤 Document Upload & Ingestion")
    st.markdown("Upload a PDF, DOCX, or TXT file to ingest into the GraphRAG system.")

    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    if uploaded and st.button("🚀 Ingest Document", type="primary"):
        with st.spinner("Processing... (extracting → chunking → embedding → graph extraction)"):
            start = time.time()
            result = api_post("ingest", files={"file": (uploaded.name, uploaded.read(), uploaded.type)})
            total_time = round(time.time() - start, 2)

        if result:
            st.success(f"✅ Ingested **{result['filename']}** — {result['total_chunks']} chunks, "
                       f"{len(result['entities'])} entities, {len(result['relations'])} relations "
                       f"in {total_time}s")

            # --- Timing Breakdown ---
            st.subheader("⏱️ Timing Breakdown")
            timing = result.get("timing", {})
            timing_df = pd.DataFrame([
                {"Step": k.replace("_s", "").replace("_", " ").title(), "Seconds": v}
                for k, v in timing.items()
            ])
            st.bar_chart(timing_df.set_index("Step"))

            # --- Token Usage ---
            st.subheader("🪙 Token Usage (Entity Extraction)")
            tok = result.get("token_usage", {})
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
                st.dataframe(ent_df, use_container_width=True)

            # --- Relations ---
            st.subheader(f"🔗 Relations ({len(result['relations'])})")
            if result["relations"]:
                rel_df = pd.DataFrame(result["relations"])
                st.dataframe(rel_df, use_container_width=True)

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
            st.markdown(f"**{len(nodes_data)}** entities, **{len(edges_data)}** relations")

            # Color map by entity type
            type_colors = {
                "Person": "#FF6B6B",
                "Organization": "#4ECDC4",
                "Concept": "#45B7D1",
                "Location": "#96CEB4",
                "Event": "#FFEAA7",
                "Policy": "#DDA0DD",
                "Technology": "#98D8C8",
                "Other": "#C9C9C9",
            }

            # Build agraph nodes and edges
            ag_nodes = []
            node_names = set()
            for n in nodes_data:
                name = n.get("name", "?")
                ntype = n.get("type", "Other")
                if name not in node_names:
                    node_names.add(name)
                    ag_nodes.append(Node(
                        id=name,
                        label=name,
                        size=25,
                        color=type_colors.get(ntype, "#C9C9C9"),
                        title=f"{name} ({ntype})",
                    ))

            ag_edges = []
            for e in edges_data:
                ag_edges.append(Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e["relation"],
                    color="#888888",
                ))

            tab_interactive, tab_svg = st.tabs(["🕸️ Interactive Graph", "🖼️ SVG Graph (Graphviz)"])

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
                # Build graphviz dot string
                dot = "digraph {\n"
                dot += '  graph [rankdir=LR, bgcolor="transparent", margin=0];\n'
                dot += '  node [style=filled, fontname="Helvetica", shape=box, rx=5, ry=5, fontsize=11];\n'
                dot += '  edge [fontname="Helvetica", fontsize=9, color="#888888"];\n'
                
                # Add nodes with colors
                for n in nodes_data:
                    name = n.get("name", "?")
                    ntype = n.get("type", "Other")
                    color = type_colors.get(ntype, "#C9C9C9")
                    dot += f'  "{name}" [fillcolor="{color}", label="{name}\\n({ntype})"];\n'
                    
                # Add edges
                for e in edges_data:
                    source = e["source"]
                    target = e["target"]
                    relation = e["relation"]
                    dot += f'  "{source}" -> "{target}" [label="{relation}"];\n'
                    
                dot += "}"
                st.graphviz_chart(dot, use_container_width=True)

            # Legend
            st.divider()
            st.subheader("🎨 Entity Type Legend")
            legend_cols = st.columns(len(type_colors))
            for i, (etype, color) in enumerate(type_colors.items()):
                legend_cols[i].markdown(
                    f'<span style="color:{color}; font-size:20px">●</span> {etype}',
                    unsafe_allow_html=True,
                )

            # Raw data
            with st.expander("📊 Raw Graph Data"):
                tab1, tab2 = st.tabs(["Nodes", "Edges"])
                with tab1:
                    st.dataframe(pd.DataFrame(nodes_data), use_container_width=True)
                with tab2:
                    st.dataframe(pd.DataFrame(edges_data), use_container_width=True)


# =============================================================================
# Page 3: Chat
# =============================================================================

elif page == "💬 Chat":
    st.title("💬 Chat with Your Documents")

    # User selector
    user_id = st.selectbox("👤 User", ["user_a", "user_b"], help="Two-user demo — separate chat histories")

    # Chat history
    history = api_get(f"chat_history/{user_id}")
    if history and history.get("history"):
        for msg in history["history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Get response
        with st.spinner("Searching documents + knowledge graph..."):
            result = api_post("chat", json={"question": question, "user_id": user_id})

        if result:
            # Show answer
            with st.chat_message("assistant"):
                st.write(result["answer"])

            # ============================================================
            # DEBUG PANEL — Full Pipeline Visibility
            # ============================================================
            debug = result.get("debug", {})

            st.divider()
            st.subheader("🔬 Pipeline Debug Panel")

            # --- Token Usage ---
            tok = debug.get("token_usage", {})
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("Prompt Tokens", tok.get("prompt_tokens", 0))
            tcol2.metric("Completion Tokens", tok.get("completion_tokens", 0))
            tcol3.metric("Total Tokens", tok.get("total", 0))

            # --- Timing ---
            timing = debug.get("timing", {})
            st.subheader("⏱️ Latency Breakdown")
            if timing:
                timing_df = pd.DataFrame([
                    {"Step": k.replace("_s", "").replace("_", " ").title(), "Seconds": v}
                    for k, v in timing.items()
                ])
                st.bar_chart(timing_df.set_index("Step"))

            # --- Vector Search Results ---
            with st.expander(f"🔷 Vector Search Results ({len(debug.get('vector_results', []))})"):
                vec_res = debug.get("vector_results", [])
                if vec_res:
                    for i, r in enumerate(vec_res):
                        st.markdown(f"**#{i+1}** — Cosine: `{r.get('vector_score', 'N/A')}` "
                                    f"| File: `{r.get('doc_filename', '')}`")
                        st.text(r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"])
                        st.divider()
                else:
                    st.info("No vector results")

            # --- BM25 Search Results ---
            with st.expander(f"🔶 BM25 Search Results ({len(debug.get('bm25_results', []))})"):
                bm25_res = debug.get("bm25_results", [])
                if bm25_res:
                    for i, r in enumerate(bm25_res):
                        st.markdown(f"**#{i+1}** — BM25 Score: `{r.get('bm25_score', 'N/A')}` "
                                    f"| File: `{r.get('doc_filename', '')}`")
                        st.text(r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"])
                        st.divider()
                else:
                    st.info("No BM25 results")

            # --- RRF Merged Results ---
            with st.expander(f"🟢 RRF Merged Results ({len(debug.get('rrf_merged', []))})"):
                rrf_res = debug.get("rrf_merged", [])
                if rrf_res:
                    rrf_df = pd.DataFrame([
                        {
                            "Rank": i + 1,
                            "RRF Score": r.get("rrf_score", ""),
                            "Vector Score": r.get("vector_score", ""),
                            "BM25 Score": r.get("bm25_score", ""),
                            "Graph Boosted": "✅" if r.get("graph_boosted") else "",
                            "Content Preview": r["content"][:100],
                        }
                        for i, r in enumerate(rrf_res)
                    ])
                    st.dataframe(rrf_df, use_container_width=True)
                else:
                    st.info("No merged results")

            # --- Graph Facts ---
            with st.expander(f"🕸️ Graph Facts ({len(debug.get('graph_facts', []))})"):
                facts = debug.get("graph_facts", [])
                if facts:
                    for f in facts:
                        st.markdown(f"**{f['source']}** →[`{f['relation']}`]→ **{f['target']}**")
                else:
                    st.info("No relevant graph facts found for this query")

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
        st.dataframe(pd.DataFrame(graph_data["nodes"]), use_container_width=True)

    # All relations
    st.subheader("🔗 All Relations")
    if graph_data and graph_data.get("edges"):
        st.dataframe(pd.DataFrame(graph_data["edges"]), use_container_width=True)

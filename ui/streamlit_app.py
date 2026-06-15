"""
GraphRAG Streamlit UI — Advanced Full Pipeline Debugging Dashboard

Pages:
1. Upload & Ingest — upload docs, see chunks/entities/relations/timing/tokens
2. Knowledge Graph — interactive graph visualization & node detail inspection
3. Chat — ask questions with full retrieval debug panel & query subgraphs
4. Pipeline Inspector — browse all chunks, entities, DB stats with search filters
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

# =============================================================================
# Configuration & Endpoints
# =============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
API = f"{BACKEND_URL}/api"

st.set_page_config(
    page_title="GraphRAG Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Premium Color Palettes & UI Themes
# =============================================================================
THEMES = {
    "🌌 Midnight Neon (Dark Mode)": {
        "bg_gradient": "linear-gradient(135deg, #0f0c29 0%, #17172e 50%, #16213e 100%)",
        "sidebar_gradient": "linear-gradient(180deg, #17172e 0%, #0f0c29 100%)",
        "text_color": "#e2e8f0",
        "card_bg": "rgba(255, 255, 255, 0.04)",
        "card_border": "1px solid rgba(0, 210, 255, 0.18)",
        "accent_color": "#00d2ff",
        "accent_hover": "#3a7bd5",
        "metric_val_gradient": "linear-gradient(135deg, #00d2ff, #3a7bd5)",
        "chat_user_bg": "rgba(0, 210, 255, 0.12)",
        "chat_user_border": "1px solid rgba(0, 210, 255, 0.3)",
        "chat_bot_bg": "rgba(255, 255, 255, 0.05)",
        "chat_bot_border": "1px solid rgba(255, 255, 255, 0.08)",
        "header_color": "#00d2ff",
        "tag_bg": "rgba(0, 210, 255, 0.15)",
        "tag_text": "#00d2ff",
        "graph_filter": "invert(0.9) hue-rotate(180deg) contrast(1.2)"
    },
    "☀️ Clean Studio (Light Mode)": {
        "bg_gradient": "linear-gradient(135deg, #f8fafc 0%, #ffffff 50%, #f1f5f9 100%)",
        "sidebar_gradient": "linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%)",
        "text_color": "#0f172a",
        "card_bg": "#ffffff",
        "card_border": "1px solid rgba(58, 123, 213, 0.25)",
        "accent_color": "#3a7bd5",
        "accent_hover": "#1d4ed8",
        "metric_val_gradient": "linear-gradient(135deg, #3a7bd5, #0f172a)",
        "chat_user_bg": "rgba(58, 123, 213, 0.08)",
        "chat_user_border": "1px solid rgba(58, 123, 213, 0.3)",
        "chat_bot_bg": "#ffffff",
        "chat_bot_border": "1px solid rgba(0, 0, 0, 0.1)",
        "header_color": "#0f172a",
        "tag_bg": "rgba(58, 123, 213, 0.12)",
        "tag_text": "#3a7bd5",
        "graph_filter": "none"
    },
    "🍇 Cyberpunk Purple": {
        "bg_gradient": "linear-gradient(135deg, #0d0118 0%, #120224 50%, #090012 100%)",
        "sidebar_gradient": "linear-gradient(180deg, #120224 0%, #0d0118 100%)",
        "text_color": "#fdf2f8",
        "card_bg": "rgba(236, 72, 153, 0.05)",
        "card_border": "1px solid rgba(236, 72, 153, 0.35)",
        "accent_color": "#ec4899",
        "accent_hover": "#db2777",
        "metric_val_gradient": "linear-gradient(135deg, #ec4899, #a855f7)",
        "chat_user_bg": "rgba(236, 72, 153, 0.15)",
        "chat_user_border": "1px solid rgba(236, 72, 153, 0.4)",
        "chat_bot_bg": "rgba(255, 255, 255, 0.02)",
        "chat_bot_border": "1px solid rgba(255, 255, 255, 0.08)",
        "header_color": "#ec4899",
        "tag_bg": "rgba(236, 72, 153, 0.18)",
        "tag_text": "#ec4899",
        "graph_filter": "invert(0.9) hue-rotate(280deg) contrast(1.25) saturate(1.6)"
    },
    "📟 Emerald Terminal": {
        "bg_gradient": "linear-gradient(135deg, #010602 0%, #030e05 50%, #010401 100%)",
        "sidebar_gradient": "linear-gradient(180deg, #030e05 0%, #010602 100%)",
        "text_color": "#22c55e",
        "card_bg": "rgba(34, 197, 94, 0.03)",
        "card_border": "1px solid rgba(34, 197, 94, 0.35)",
        "accent_color": "#22c55e",
        "accent_hover": "#15803d",
        "metric_val_gradient": "linear-gradient(135deg, #22c55e, #166534)",
        "chat_user_bg": "rgba(34, 197, 94, 0.12)",
        "chat_user_border": "1px solid rgba(34, 197, 94, 0.45)",
        "chat_bot_bg": "rgba(255, 255, 255, 0.01)",
        "chat_bot_border": "1px solid rgba(34, 197, 94, 0.18)",
        "header_color": "#22c55e",
        "tag_bg": "rgba(34, 197, 94, 0.18)",
        "tag_text": "#22c55e",
        "graph_filter": "invert(0.9) hue-rotate(85deg) contrast(1.25) saturate(2)"
    }
}

def inject_custom_css(theme_name):
    t = THEMES[theme_name]
    st.markdown(f"""
    <style>
    :root {{
        --bg-grad: {t['bg_gradient']};
        --side-grad: {t['sidebar_gradient']};
        --text-col: {t['text_color']};
        --card-bg: {t['card_bg']};
        --card-border: {t['card_border']};
        --accent-col: {t['accent_color']};
        --accent-hov: {t['accent_hover']};
        --metric-val-grad: {t['metric_val_gradient']};
        --chat-u-bg: {t['chat_user_bg']};
        --chat-u-border: {t['chat_user_border']};
        --chat-b-bg: {t['chat_bot_bg']};
        --chat-b-border: {t['chat_bot_border']};
        --header-col: {t['header_color']};
        --tag-bg: {t['tag_bg']};
        --tag-text: {t['tag_text']};
        --graph-filter: {t['graph_filter']};
    }}
    
    [data-testid="stAppViewContainer"] {{
        background: var(--bg-grad) !important;
        color: var(--text-col) !important;
    }}
    [data-testid="stSidebar"] {{
        background: var(--side-grad) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }}
    
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, div, span, label, td, th {{
        color: var(--text-col) !important;
    }}
    
    /* --- Metrics cards --- */
    [data-testid="stMetric"] {{
        background: var(--card-bg) !important;
        border: var(--card-border) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }}
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        background: var(--metric-val-grad) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 700 !important;
    }}
    
    /* --- Cards for Document lists --- */
    .doc-card {{
        background: var(--card-bg);
        border: var(--card-border);
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }}
    .doc-card:hover {{
        border-color: var(--accent-col);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }}
    
    /* --- Custom pipeline step cards --- */
    .pipeline-step {{
        background: var(--card-bg);
        border: var(--card-border);
        border-left: 4px solid var(--accent-col) !important;
        border-radius: 4px 12px 12px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }}
    .pipeline-step-done {{
        border-left-color: #00c853 !important;
    }}
    .step-label {{
        color: var(--accent-col);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }}
    .step-value {{
        color: var(--text-col);
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 4px;
    }}
    
    /* --- Graph fact badges --- */
    .graph-fact {{
        background: var(--card-bg);
        border: var(--card-border);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.95rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    
    /* --- Chat Bubbles --- */
    .chat-bubble {{
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 85%;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        line-height: 1.5;
        font-size: 1rem;
        transition: all 0.3s ease;
    }}
    .chat-user {{
        background: var(--chat-u-bg) !important;
        border: var(--chat-u-border) !important;
        color: var(--text-col) !important;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }}
    .chat-bot {{
        background: var(--chat-b-bg) !important;
        border: var(--chat-b-border) !important;
        color: var(--text-col) !important;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }}
    
    /* --- Progress bars --- */
    .score-bar-container {{
        width: 100%;
        background-color: rgba(255,255,255,0.05);
        border-radius: 6px;
        margin-top: 5px;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .score-bar {{
        height: 10px;
        background-color: var(--accent-col);
        border-radius: 6px;
    }}
    
    /* --- Stepper --- */
    .stepper {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 25px;
        background: var(--card-bg);
        border: var(--card-border);
        border-radius: 12px;
        padding: 20px;
    }}
    .step {{
        text-align: center;
        width: 23%;
        position: relative;
    }}
    .step-num {{
        width: 30px;
        height: 30px;
        line-height: 28px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
        border: 2px solid rgba(255,255,255,0.2);
        color: var(--text-col);
        margin: 0 auto 8px auto;
        font-weight: bold;
        font-size: 0.9rem;
    }}
    .step-active .step-num {{
        background: var(--accent-col);
        border-color: var(--accent-col);
        color: #000;
        box-shadow: 0 0 10px var(--accent-col);
    }}
    .step-done .step-num {{
        background: #00c853;
        border-color: #00c853;
        color: #fff;
    }}
    .step-text {{
        font-size: 0.8rem;
        font-weight: 500;
    }}
    
    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: rgba(0,0,0,0.1);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--accent-col);
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent-hov);
    }}
    
    /* --- Streamlit Element Overrides (Selectboxes, textareas, tags, buttons) --- */
    div[data-baseweb="select"] {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] div {{
        color: var(--text-col) !important;
    }}
    
    /* Active Selected Tags in Multiselect */
    span[data-baseweb="tag"] {{
        background-color: var(--tag-bg) !important;
        color: var(--tag-text) !important;
        border: 1px solid var(--accent-col) !important;
        border-radius: 4px !important;
    }}
    span[data-baseweb="tag"] span {{
        color: var(--tag-text) !important;
    }}
    span[data-baseweb="tag"] svg {{
        fill: var(--tag-text) !important;
    }}
    
    /* Dropdown Options List */
    div[data-baseweb="popover"] ul, ul[role="listbox"] {{
        background-color: var(--side-grad) !important;
        border: var(--card-border) !important;
    }}
    li[role="option"] {{
        color: var(--text-col) !important;
    }}
    li[role="option"]:hover, li[data-highlighted="true"] {{
        background-color: var(--accent-col) !important;
        color: #000000 !important;
    }}
    
    /* Chat Input Container */
    div[data-testid="stChatInput"] {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        border-radius: 24px !important;
        padding: 4px 12px !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: var(--text-col) !important;
    }}
    div[data-testid="stChatInput"] button {{
        background-color: var(--accent-col) !important;
        color: #000000 !important;
        border-radius: 50% !important;
    }}
    
    /* Text Inputs, Selectboxes, Textareas */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        color: var(--text-col) !important;
        border-radius: 8px !important;
    }}
    
    .streamlit-expanderHeader {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        border-radius: 8px !important;
        color: var(--text-col) !important;
    }}
    
    .stButton > button {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        color: var(--text-col) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background-color: var(--accent-col) !important;
        color: #000000 !important;
        border-color: var(--accent-col) !important;
        box-shadow: 0 0 10px var(--accent-col) !important;
    }}
    
    /* --- File Uploader Overrides --- */
    [data-testid="stFileUploader"] {{
        background-color: var(--card-bg) !important;
        border: var(--card-border) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: transparent !important;
        border: 1px dashed var(--accent-col) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] label {{
        color: var(--text-col) !important;
        font-weight: 600 !important;
    }}
    
    /* --- Graph Iframe Alignment & Transparency Filter --- */
    iframe[title*="streamlit_agraph"] {{
        background-color: transparent !important;
        background: transparent !important;
        border: var(--card-border) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
        filter: var(--graph-filter) !important;
    }}
    
    iframe {{
        background-color: transparent !important;
        background: transparent !important;
        filter: var(--graph-filter) !important;
    }}
    
    </style>
    """, unsafe_allow_html=True)



# =============================================================================
# API Helper Functions
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

# =============================================================================
# Entity type color map
# =============================================================================
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
    """Render entity type color legend as inline flex layout."""
    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; margin-bottom: 15px;">'
    for etype, color in TYPE_COLORS.items():
        legend_html += (
            f'<div style="display: flex; align-items: center; font-size: 0.9rem; font-weight: 500;">'
            f'  <span style="color:{color}; font-size: 1.25rem; margin-right: 6px;">●</span>{etype}'
            f'</div>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)


# =============================================================================
# Sidebar — Navigation + System Status & Theme Engine
# =============================================================================
st.sidebar.title("🔬 GraphRAG Studio")
st.sidebar.caption("Pure Graph-Enhanced Retrieval-Augmented Generation")

# Theme selection
selected_theme = st.sidebar.selectbox("🎨 UI Theme Preset", list(THEMES.keys()), index=0)
inject_custom_css(selected_theme)

# Navigation
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
    with st.sidebar.expander("⚙️ Backend Config"):
        st.json(health.get("config", {}))

# Reset button
st.sidebar.divider()
if st.sidebar.button("🗑️ Reset All Data", type="secondary", use_container_width=True, help="Clear all documents, chunks, and graph"):
    result = api_delete("reset")
    if result:
        st.sidebar.success("All data cleared successfully!")
        time.sleep(1.0)
        st.rerun()

# =============================================================================
# Page 1: Upload & Ingest
# =============================================================================
if page == "📤 Upload & Ingest":
    st.title("📤 Document Ingestion Hub")
    st.markdown("Upload documents (PDF, DOCX, TXT) to split them into vector chunks and extract entities and relations into the graph.")

    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    if uploaded and st.button("🚀 Ingest Document", type="primary"):
        stepper_placeholder = st.empty()
        
        def render_stepper(active_step):
            steps_html = '<div class="stepper">'
            labels = ["Extract Text", "Semantic Chunking", "Generate Embeddings", "Extract Graph"]
            for i, label in enumerate(labels, 1):
                step_class = "step"
                if i < active_step:
                    step_class += " step-done"
                elif i == active_step:
                    step_class += " step-active"
                steps_html += (
                    f'<div class="{step_class}">'
                    f'  <div class="step-num">{i if i >= active_step else "✓"}</div>'
                    f'  <div class="step-text">{label}</div>'
                    f'</div>'
                )
            steps_html += '</div>'
            stepper_placeholder.markdown(steps_html, unsafe_allow_html=True)

        render_stepper(1)
        time.sleep(0.3)
        render_stepper(2)
        time.sleep(0.3)
        render_stepper(3)
        time.sleep(0.3)
        render_stepper(4)

        with st.spinner("Extracting entities & storing graph relationships (running LLM)..."):
            start = time.time()
            result = api_post("ingest", files={"file": (uploaded.name, uploaded.read(), uploaded.type)})
            total_time = round(time.time() - start, 2)

        if result:
            render_stepper(5)
            st.success(f"✅ Ingested **{result['filename']}** — {result['total_chunks']} chunks, "
                       f"{len(result['entities'])} entities, {len(result['relations'])} relations "
                       f"in {total_time}s")

            # Pipeline metrics cards
            st.subheader("📊 Ingestion Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📦 Chunks", result['total_chunks'])
            col2.metric("🏷️ Entities", len(result['entities']))
            col3.metric("🔗 Relations", len(result['relations']))
            tok = result.get("token_usage", {})
            col4.metric("🪙 Total Tokens", tok.get("total", 0))

            # Timings
            st.subheader("⏱️ Ingestion Pipeline Timing")
            timing = result.get("timing", {})
            timing_df = pd.DataFrame([
                {"Step": k.replace("_s", "").replace("_", " ").title(), "Seconds": v}
                for k, v in timing.items()
            ])
            st.bar_chart(timing_df.set_index("Step"))

            # Chunks
            st.subheader(f"📦 Chunks Preview ({result['total_chunks']})")
            for chunk in result["chunks"]:
                with st.expander(f"Chunk {chunk['chunk_index']} — {len(chunk['content'])} chars"):
                    st.text(chunk["content"])
                    st.caption(f"Embedding preview: {chunk['embedding_preview']}")

            # Entities
            st.subheader(f"🏷️ Extracted Entities ({len(result['entities'])})")
            if result["entities"]:
                st.dataframe(pd.DataFrame(result["entities"]), use_container_width=True, hide_index=True)

            # Relations
            st.subheader(f"🔗 Extracted Relations ({len(result['relations'])})")
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

    # List of Ingested Documents
    st.divider()
    st.subheader("📚 Ingested Documents")
    docs = api_get("documents")
    if docs and docs.get("documents"):
        for doc in docs["documents"]:
            doc_id = doc["id"]
            filename = doc["filename"]
            chunks_count = doc["chunk_count"]
            created = doc["created_at"]
            
            col_info, col_del = st.columns([7, 1])
            with col_info:
                st.markdown(
                    f'<div class="doc-card">'
                    f'  <div>'
                    f'    <span style="font-size: 1.1rem; font-weight: bold; color: var(--accent-col);">📄 {filename}</span><br/>'
                    f'    <span style="font-size: 0.85rem; color: #888;">{chunks_count} chunks | Ingested {created}</span>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_del:
                if st.button("🗑️", key=f"del_doc_{doc_id}", help=f"Delete {filename}", use_container_width=True):
                    del_result = api_delete(f"documents/{doc_id}")
                    if del_result:
                        st.success(f"Deleted {filename}!")
                        time.sleep(1.0)
                        st.rerun()
    else:
        st.info("No documents ingested yet. Upload one above!")

# =============================================================================
# Page 2: Knowledge Graph
# =============================================================================
elif page == "🕸️ Knowledge Graph":
    st.title("🕸️ Knowledge Graph Studio")
    st.markdown("Explore and inspect the entities and relationships in the Apache AGE graph database.")

    graph_data = api_get("graph")
    if graph_data:
        nodes_data = graph_data.get("nodes", [])
        edges_data = graph_data.get("edges", [])

        if not nodes_data:
            st.info("No entities in the knowledge graph yet. Ingest a document first!")
        else:
            # Layout options
            layout_type = st.radio("Graph Layout", ["Forced-Directed", "Hierarchical", "Circular"], horizontal=True)
            
            # Legend on its own full-width row
            render_legend()
            
            st.divider()

            # Filters
            col_types, col_search = st.columns([2, 1])
            all_types = sorted(set(n.get("type", "Other") for n in nodes_data))
            selected_types = col_types.multiselect(
                "Filter by entity type",
                all_types,
                default=all_types,
            )
            search_query = col_search.text_input("🔍 Search & Highlight Entity", "")

            # Filter data
            filtered_nodes = [n for n in nodes_data if n.get("type", "Other") in selected_types]
            filtered_names = set(n.get("name") for n in filtered_nodes)
            filtered_edges = [
                e for e in edges_data
                if e["source"] in filtered_names and e["target"] in filtered_names
            ]

            # Build nodes/edges for agraph
            ag_nodes = []
            node_names = set()
            for n in filtered_nodes:
                name = n.get("name", "?")
                ntype = n.get("type", "Other")
                
                # Check for highlight search match
                is_match = search_query and search_query.lower() in name.lower()
                size = 35 if is_match else 22
                color = "#ff007f" if is_match else TYPE_COLORS.get(ntype, "#C9C9C9")
                
                if name not in node_names:
                    node_names.add(name)
                    ag_nodes.append(Node(
                        id=name,
                        label=name,
                        size=size,
                        color=color,
                        title=f"{name} ({ntype})",
                    ))

            ag_edges = [
                Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e["relation"],
                    color="#4ade80" if (search_query and (search_query.lower() in e["source"].lower() or search_query.lower() in e["target"].lower())) else "#888888",
                )
                for e in filtered_edges
            ]

            col_canvas, col_details = st.columns([3, 1])

            with col_canvas:
                config = Config(
                    width=950,
                    height=650,
                    directed=True,
                    physics=(layout_type == "Forced-Directed"),
                    hierarchical=(layout_type == "Hierarchical"),
                    nodeHighlightBehavior=True,
                    highlightColor="#ff007f",
                    background="rgba(0,0,0,0)"
                )
                clicked_node = agraph(nodes=ag_nodes, edges=ag_edges, config=config)

            with col_details:
                st.subheader("🔍 Entity Inspector")
                if clicked_node:
                    node_props = next((n for n in nodes_data if n.get("name") == clicked_node), None)
                    ntype = node_props.get("type", "Other") if node_props else "Other"
                    color = TYPE_COLORS.get(ntype, "#C9C9C9")
                    
                    st.markdown(
                        f'<div style="padding:15px; border-radius:10px; border: 1px solid {color}; background:rgba(255,255,255,0.03);">'
                        f'  <span style="font-size:1.3rem; font-weight:bold; color:{color};">● {clicked_node}</span><br/>'
                        f'  <span style="font-size:0.9rem; color:#aaa;">Type: {ntype}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### Connections")
                    incoming = [e for e in edges_data if e["target"] == clicked_node]
                    outgoing = [e for e in edges_data if e["source"] == clicked_node]
                    
                    if not incoming and not outgoing:
                        st.info("No connections for this node.")
                    
                    if outgoing:
                        st.markdown("**Outgoing relations:**")
                        for e in outgoing:
                            st.write(f"👉 `{e['relation']}` ➔ **{e['target']}**")
                    if incoming:
                        st.markdown("**Incoming relations:**")
                        for e in incoming:
                            st.write(f"👈 **{e['source']}** ➔ `{e['relation']}`")
                else:
                    st.info("💡 Click on any node in the graph canvas to inspect its adjacent nodes and attributes.")

            # Raw Graph Table representation
            with st.expander("📊 Raw Graph Data Table"):
                tab1, tab2 = st.tabs(["Nodes List", "Edges List"])
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
    st.title("💬 Chat Playground")
    st.markdown("Ask questions and watch the pure GraphRAG pipeline execute step-by-step.")

    # User Selection & Clear Buttons
    col_user, col_clear = st.columns([6, 1])
    user_id = col_user.selectbox("👤 User Session", ["user_a", "user_b"], help="Separate histories for different users")
    
    if col_clear.button("🧹 Clear Chat", use_container_width=True, help="Clear history for this user"):
        clear_res = api_delete(f"chat_history/{user_id}")
        if clear_res:
            st.success("Chat history cleared!")
            time.sleep(1.0)
            st.rerun()

    # Chat history representation
    history = api_get(f"chat_history/{user_id}")
    if history and history.get("history"):
        for msg in history["history"]:
            role = msg["role"]
            content = msg["content"]
            bubble_class = "chat-user" if role == "user" else "chat-bot"
            align_right = "text-align: right;" if role == "user" else ""
            avatar = "👤" if role == "user" else "🤖"
            st.markdown(
                f'<div style="{align_right}">'
                f'  <div class="chat-bubble {bubble_class}" style="display: inline-block; text-align: left;">'
                f'    <strong>{avatar} {role.capitalize()}:</strong><br/>{content}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.markdown(
            f'<div style="text-align: right;">'
            f'  <div class="chat-bubble chat-user" style="display: inline-block; text-align: left;">'
            f'    <strong>👤 User:</strong><br/>{question}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Executing GraphRAG retrieval pipeline (Embedding + Vector Search + Graph Match)..."):
            result = api_post("chat", json={"question": question, "user_id": user_id})

        if result:
            st.markdown(
                f'<div style="text-align: left;">'
                f'  <div class="chat-bubble chat-bot" style="display: inline-block; text-align: left;">'
                f'    <strong>🤖 GraphRAG:</strong><br/>{result["answer"]}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # =================================================================
            # PIPELINE DEBUGGING PANEL
            # =================================================================
            debug = result.get("debug", {})

            st.divider()
            st.subheader("🔬 GraphRAG Pipeline Debug Panel")

            # 1. Pipeline Stepper timings
            timing = debug.get("timing", {})
            steps = [
                ("1️⃣ Embed Query", "embed_query_s", "Convert text to vector"),
                ("2️⃣ Vector Search", "vector_search_s", f"Found {len(debug.get('vector_results', []))} chunks"),
                ("3️⃣ Graph Search", "graph_search_s", f"Found {len(debug.get('graph_facts', []))} facts"),
                ("4️⃣ Generation", "generation_s", "Synthesized response"),
            ]

            cols = st.columns(4)
            for i, (label, key, desc) in enumerate(steps):
                t = timing.get(key, 0)
                cols[i].markdown(
                    f'<div class="pipeline-step pipeline-step-done">'
                    f'  <div class="step-label">{label}</div>'
                    f'  <div class="step-value">{t}s</div>'
                    f'  <div style="color:#aaa; font-size:0.8rem; margin-top:2px;">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 2. Token details
            st.markdown("#### 🪙 Model Token Metrics")
            tok = debug.get("token_usage", {})
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("Prompt Tokens", tok.get("prompt_tokens", 0))
            tcol2.metric("Completion Tokens", tok.get("completion_tokens", 0))
            tcol3.metric("Total Tokens", tok.get("total", 0))

            # 3. Question Entities
            q_entities = debug.get("question_entities", [])
            if q_entities:
                st.markdown("#### 🏷️ Extracted Search Entities")
                st.markdown(" ".join(f'`{e}`' for e in q_entities))

            # 4. Matched facts subgraph & badges
            st.subheader("🕸️ Query Subgraph Facts")
            facts = debug.get("graph_facts", [])
            if facts:
                tab_badge, tab_subgraph = st.tabs(["🏷️ Fact Badges", "🕸️ Interactive Subgraph"])
                with tab_badge:
                    for f in facts:
                        st.markdown(
                            f'<div class="graph-fact">'
                            f'  <strong>{f["source"]}</strong> '
                            f'  → <code>{f["relation"]}</code> → '
                            f'  <strong>{f["target"]}</strong>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                with tab_subgraph:
                    sub_nodes = []
                    sub_edges = []
                    sub_names = set()
                    
                    # Resolve node types from general graph cache if possible
                    cached_nodes = {}
                    graph_all = api_get("graph")
                    if graph_all:
                        for n in graph_all.get("nodes", []):
                            cached_nodes[n["name"]] = n.get("type", "Other")

                    for f in facts:
                        for name in [f["source"], f["target"]]:
                            if name not in sub_names:
                                sub_names.add(name)
                                ntype = cached_nodes.get(name, "Other")
                                sub_nodes.append(Node(
                                    id=name,
                                    label=name,
                                    size=20,
                                    color=TYPE_COLORS.get(ntype, "#C9C9C9"),
                                    title=f"{name} ({ntype})"
                                ))
                        sub_edges.append(Edge(
                            source=f["source"],
                            target=f["target"],
                            label=f["relation"],
                            color="#3a7bd5"
                        ))
                    
                    sub_config = Config(
                        width=900,
                        height=350,
                        directed=True,
                        physics=True,
                        hierarchical=False,
                        nodeHighlightBehavior=True,
                        background="rgba(0,0,0,0)"
                    )
                    agraph(nodes=sub_nodes, edges=sub_edges, config=sub_config)
            else:
                st.info("No relevant graph facts found for this query.")

            # 5. Vector Results with Similarity Score Progress Bars
            st.subheader("🔷 Vector Search Results")
            with st.expander("Inspect Similar Document Chunks"):
                vec_res = debug.get("vector_results", [])
                if vec_res:
                    for i, r in enumerate(vec_res):
                        boost = " 🔗 **Graph-boosted**" if r.get("graph_boosted") else ""
                        score = r.get("vector_score", 0.0)
                        score_pct = int(score * 100) if score else 0
                        st.markdown(f"**#{i+1}** — Score: `{score}`{boost} | File: `{r.get('doc_filename', '')}`")
                        st.markdown(
                            f'<div class="score-bar-container">'
                            f'  <div class="score-bar" style="width:{score_pct}%;"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.text(r["content"])
                        st.divider()
                else:
                    st.info("No vector results found.")

            # 6. Final Sources
            st.subheader("📄 Context Sources Fed to LLM")
            with st.expander("View Injected Context"):
                for i, s in enumerate(result.get("sources", [])):
                    boost = " 🔗 Graph-boosted" if s.get("graph_boosted") else ""
                    st.markdown(f"**Source {i+1}** (Chunk {s['chunk_id']}) from `{s['doc_filename']}`{boost}")
                    st.text(s["content"])
                    st.divider()

# =============================================================================
# Page 4: Pipeline Inspector
# =============================================================================
elif page == "🔍 Pipeline Inspector":
    st.title("🔍 Database & Pipeline Inspector")
    st.markdown("Browse and filter the underlying database tables containing chunks, entities, and relations.")

    # Get data
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

    tab_chunks, tab_entities, tab_relations = st.tabs(["📦 Chunks Explorer", "🏷️ Entities Table", "🔗 Relations Table"])

    with tab_chunks:
        st.subheader("Explore Chunks")
        if docs and docs.get("documents"):
            doc_names = {d["id"]: d["filename"] for d in docs["documents"]}
            selected_doc = st.selectbox(
                "Select document",
                options=list(doc_names.keys()),
                format_func=lambda x: doc_names[x],
                key="inspector_doc_select"
            )
            
            chunk_search = st.text_input("🔍 Filter text content", "", key="chunk_search_input")
            
            if selected_doc:
                chunks_res = api_get(f"chunks/{selected_doc}")
                if chunks_res and chunks_res.get("chunks"):
                    filtered_chunks = chunks_res["chunks"]
                    if chunk_search:
                        filtered_chunks = [c for c in filtered_chunks if chunk_search.lower() in c["content"].lower()]
                    
                    st.write(f"Showing {len(filtered_chunks)} chunks:")
                    for c in filtered_chunks:
                        with st.expander(f"Chunk {c['chunk_index']} — {len(c['content'])} characters"):
                            st.text(c["content"])
        else:
            st.info("No documents ingested yet.")

    with tab_entities:
        st.subheader("All Extracted Entities")
        if graph_data and graph_data.get("nodes"):
            nodes_df = pd.DataFrame(graph_data["nodes"])
            
            col_search, col_type = st.columns(2)
            ent_search = col_search.text_input("🔍 Search Entity Name", "", key="ent_search_input")
            all_types = sorted(list(set(nodes_df["type"])))
            ent_type = col_type.multiselect("Filter by Type", all_types, default=all_types, key="ent_type_filter")
            
            filtered_nodes = nodes_df
            if ent_search:
                filtered_nodes = filtered_nodes[filtered_nodes["name"].str.contains(ent_search, case=False)]
            if ent_type:
                filtered_nodes = filtered_nodes[filtered_nodes["type"].isin(ent_type)]
                
            st.write(f"Showing {len(filtered_nodes)} entities:")
            st.dataframe(filtered_nodes, use_container_width=True, hide_index=True)
        else:
            st.info("No entities in database yet.")

    with tab_relations:
        st.subheader("All Extracted Relations")
        if graph_data and graph_data.get("edges"):
            edges_df = pd.DataFrame(graph_data["edges"])
            
            col_src, col_rel = st.columns(2)
            rel_src_search = col_src.text_input("🔍 Search Source / Target Name", "", key="rel_src_search_input")
            rel_type_search = col_rel.text_input("🔍 Search Relation Type (e.g. works_at)", "", key="rel_type_search_input")
            
            filtered_edges = edges_df
            if rel_src_search:
                filtered_edges = filtered_edges[
                    filtered_edges["source"].str.contains(rel_src_search, case=False) | 
                    filtered_edges["target"].str.contains(rel_src_search, case=False)
                ]
            if rel_type_search:
                filtered_edges = filtered_edges[filtered_edges["relation"].str.contains(rel_type_search, case=False)]
                
            st.write(f"Showing {len(filtered_edges)} relations:")
            st.dataframe(filtered_edges, use_container_width=True, hide_index=True)
        else:
            st.info("No relations in database yet.")

"""
FastAPI application — main entry point.
Mounts ingestion and retrieval routers, provides health + utility endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import check_db_health, execute_sql, execute_cypher, get_conn
from app.llm_client import check_llm_health, warmup_embed_model
from app.ingestion import router as ingest_router
from app.retrieval import router as retrieval_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# App Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify DB connection, warm up embedding model. Shutdown: cleanup."""
    logger.info("Starting GraphRAG backend...")
    logger.info(f"Database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'}")
    logger.info(f"LLM: {settings.llm_model} at {settings.vllm_base_url}")
    logger.info(f"Embedding: {settings.embedding_model} (CPU)")

    if check_db_health():
        logger.info("✓ Database connected, extensions loaded (pgvector + AGE)")
    else:
        logger.warning("✗ Database not ready — will retry on first request")

    # Warm up embedding model so first request is fast
    try:
        warmup_embed_model()
    except Exception as e:
        logger.warning(f"Embedding warmup failed (will retry on first use): {e}")

    yield
    logger.info("Shutting down GraphRAG backend.")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="GraphRAG API",
    description="Graph-enhanced RAG with pgvector + Apache AGE + vLLM",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(ingest_router)
app.include_router(retrieval_router)


# =============================================================================
# Utility Endpoints
# =============================================================================

@app.get("/api/health")
async def health():
    """Health check — reports status of DB and LLM connections."""
    db_ok = check_db_health()
    llm_ok = check_llm_health()
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "connected" if db_ok else "disconnected",
        "llm": "connected" if llm_ok else "disconnected",
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "top_k": settings.top_k,
        },
    }


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    rows = execute_sql(
        "SELECT d.id, d.filename, d.created_at, COUNT(c.id) as chunk_count "
        "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
        "GROUP BY d.id ORDER BY d.created_at DESC"
    )
    return {"documents": rows}


@app.get("/api/chunks/{doc_id}")
async def get_chunks(doc_id: int):
    """Get all chunks for a document (for pipeline inspection)."""
    rows = execute_sql(
        "SELECT id, chunk_index, content FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
        (doc_id,),
    )
    return {"chunks": rows}


@app.get("/api/graph")
async def get_graph():
    """Get all nodes and edges from the knowledge graph (for visualization)."""
    # Get all entity nodes
    try:
        nodes_raw = execute_cypher(
            "MATCH (n:Entity) RETURN n",
            columns="v agtype",
        )
    except Exception:
        nodes_raw = []

    nodes = []
    for n in nodes_raw:
        if isinstance(n, dict) and "properties" in n:
            nodes.append({
                "id": n.get("id", id(n)),
                "name": n["properties"].get("name", "?"),
                "type": n["properties"].get("type", "Other"),
            })
        elif isinstance(n, dict):
            nodes.append({
                "id": id(n),
                "name": n.get("name", "?"),
                "type": n.get("type", "Other"),
            })

    # Get all edges
    try:
        edges_raw = execute_cypher(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.name, r.type, b.name",
            columns="s agtype, r agtype, t agtype",
        )
    except Exception:
        edges_raw = []

    edges = []
    for row in edges_raw:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            edges.append({
                "source": str(row[0]).strip('"'),
                "relation": str(row[1]).strip('"'),
                "target": str(row[2]).strip('"'),
            })

    return {"nodes": nodes, "edges": edges}


@app.get("/api/chat_history/{user_id}")
async def get_chat_history(user_id: str):
    """Get chat history for a specific user."""
    rows = execute_sql(
        "SELECT role, content, created_at FROM chat_history "
        "WHERE user_id = %s ORDER BY created_at ASC",
        (user_id,),
    )
    return {"history": rows}


@app.delete("/api/reset")
async def reset_all():
    """Delete all data — documents, chunks, chat history, and graph. For demo use."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history")
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM documents")
        conn.commit()

    # Clear graph
    try:
        execute_cypher("MATCH (n) DETACH DELETE n", columns="v agtype")
    except Exception:
        pass  # Graph might be empty

    logger.info("All data reset")
    return {"status": "ok", "message": "All data deleted"}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a single document and its associated chunks (via cascade)."""
    execute_sql("DELETE FROM documents WHERE id = %s", (doc_id,))
    logger.info(f"Document {doc_id} deleted")
    return {"status": "ok", "message": f"Document {doc_id} deleted"}


@app.delete("/api/chat_history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user."""
    execute_sql("DELETE FROM chat_history WHERE user_id = %s", (user_id,))
    logger.info(f"Chat history for user '{user_id}' cleared")
    return {"status": "ok", "message": f"Chat history for user '{user_id}' cleared"}


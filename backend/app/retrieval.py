"""
Retrieval pipeline — hybrid search with full pipeline debugging:
1. Vector search (pgvector cosine similarity)
2. BM25 search (PostgreSQL full-text search with ts_rank)
3. Graph search (Apache AGE — find related entities/facts)
4. RRF fusion (merge vector + BM25 rankings)
5. Graph boosting (boost chunks that mention graph-connected entities)
6. Answer generation (LLM with merged context)

Every step's results and scores are returned for frontend display.
"""

import json
import logging
import time

from fastapi import APIRouter
from rank_bm25 import BM25Okapi

from app.config import settings
from app.database import execute_sql, execute_cypher, get_conn
from app.llm_client import embed_single, generate
from app.prompts import ANSWER_SYSTEM, ANSWER_PROMPT, QUESTION_ENTITY_PROMPT
from app.models import (
    ChatRequest, ChatResponse, RetrievedChunk,
    GraphFact, PipelineDebug,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["retrieval"])


# =============================================================================
# Vector Search (pgvector)
# =============================================================================

def vector_search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Find most similar chunks using cosine distance in pgvector."""
    sql = """
        SELECT c.id, c.content, c.chunk_index, d.filename,
               1 - (c.embedding <=> %s::vector) AS cosine_score
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    emb_str = str(query_embedding)
    return execute_sql(sql, (emb_str, emb_str, top_k))


# =============================================================================
# BM25 Search (PostgreSQL full-text search)
# =============================================================================

def bm25_search(query: str, top_k: int) -> list[dict]:
    """Full-text search using PostgreSQL's built-in ts_rank (BM25-style scoring)."""
    sql = """
        SELECT c.id, c.content, c.chunk_index, d.filename,
               ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS bm25_score
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.tsv @@ plainto_tsquery('english', %s)
        ORDER BY bm25_score DESC
        LIMIT %s
    """
    return execute_sql(sql, (query, query, top_k))


# =============================================================================
# Graph Search (Apache AGE)
# =============================================================================

def extract_question_entities(question: str) -> list[str]:
    """Use LLM to extract searchable entity names from the question."""
    prompt = QUESTION_ENTITY_PROMPT.format(question=question)
    result = generate(prompt, max_tokens=1024)
    try:
        from app.llm_client import parse_json_from_llm
        entities = parse_json_from_llm(result["text"])
        return entities if isinstance(entities, list) else []
    except Exception:
        # Fallback: split question into significant words
        stop = {"what", "how", "why", "when", "where", "who", "is", "are", "the", "a", "an",
                "in", "on", "of", "for", "to", "and", "or", "do", "does", "can", "will"}
        return [w for w in question.lower().split() if w not in stop and len(w) > 2]


def graph_search(entities: list[str]) -> list[dict]:
    """Find facts in the knowledge graph related to the given entities."""
    facts = []
    for entity in entities:
        name = entity.replace("'", "\\'")
        # Find all relations where this entity is source or target
        cypher = (
            f"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            f"WHERE a.name =~ '(?i).*{name}.*' OR b.name =~ '(?i).*{name}.*' "
            f"RETURN a.name, r.type, b.name"
        )
        try:
            rows = execute_cypher(cypher, columns="s agtype, r agtype, t agtype")
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    facts.append({
                        "source": str(row[0]).strip('"'),
                        "relation": str(row[1]).strip('"'),
                        "target": str(row[2]).strip('"'),
                    })
                elif isinstance(row, dict):
                    facts.append(row)
        except Exception as e:
            logger.warning(f"Graph search failed for entity '{entity}': {e}")
    
    # Deduplicate
    seen = set()
    unique = []
    for f in facts:
        key = (f["source"], f["relation"], f["target"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# =============================================================================
# Reciprocal Rank Fusion (RRF)
# =============================================================================

def rrf_merge(vector_results: list[dict], bm25_results: list[dict],
              k: int = 60) -> list[dict]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across all lists.
    Uses rank position, not raw scores — handles different score scales.
    """
    scores = {}      # chunk_id -> rrf_score
    chunk_map = {}   # chunk_id -> chunk data

    for rank, r in enumerate(vector_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = r

    for rank, r in enumerate(bm25_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = r

    # Sort by RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, rrf_score in ranked:
        data = chunk_map[cid]
        data["rrf_score"] = round(rrf_score, 6)
        results.append(data)

    return results


# =============================================================================
# Chat Endpoint
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Hybrid GraphRAG retrieval + answer generation.
    Returns answer with full pipeline debug info.
    """
    timing = {}
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    # 1. Embed the question
    t0 = time.time()
    q_embedding = embed_single(req.question)
    timing["embed_query_s"] = round(time.time() - t0, 3)

    # 2. Vector search
    t0 = time.time()
    vec_results = vector_search(q_embedding, settings.top_k)
    timing["vector_search_s"] = round(time.time() - t0, 3)

    # 3. BM25 search
    t0 = time.time()
    bm25_results = bm25_search(req.question, settings.top_k)
    timing["bm25_search_s"] = round(time.time() - t0, 3)

    # 4. Graph search
    t0 = time.time()
    q_entities = extract_question_entities(req.question)
    graph_facts = graph_search(q_entities)
    timing["graph_search_s"] = round(time.time() - t0, 3)

    # 5. RRF Fusion
    t0 = time.time()
    merged = rrf_merge(vec_results, bm25_results, k=settings.rrf_k)
    timing["rrf_merge_s"] = round(time.time() - t0, 3)

    # 6. Graph boosting — mark chunks that contain graph-connected entities
    graph_entity_names = set()
    for f in graph_facts:
        graph_entity_names.add(f["source"].lower())
        graph_entity_names.add(f["target"].lower())

    for item in merged:
        content_lower = item["content"].lower()
        item["graph_boosted"] = any(e in content_lower for e in graph_entity_names)

    # Take top K for context
    top_chunks = merged[: settings.top_k]

    # 7. Build context and generate answer
    t0 = time.time()
    context_str = "\n\n".join(
        f"[Chunk {i}] (from {c['filename']}):\n{c['content']}"
        for i, c in enumerate(top_chunks)
    )
    graph_str = "\n".join(
        f"- {f['source']} --[{f['relation']}]--> {f['target']}"
        for f in graph_facts
    ) if graph_facts else "No relevant graph facts found."

    prompt = ANSWER_PROMPT.format(
        context_chunks=context_str,
        graph_facts=graph_str,
        question=req.question,
    )
    answer_result = generate(prompt, system=ANSWER_SYSTEM, max_tokens=2048)
    total_tokens["prompt_tokens"] += answer_result["prompt_tokens"]
    total_tokens["completion_tokens"] += answer_result["completion_tokens"]
    timing["generation_s"] = answer_result["latency_s"]

    total_tokens["total"] = total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]

    # 8. Save chat history
    _save_chat(req.user_id, req.question, answer_result["text"])

    # 9. Build response with full debug info
    def _to_retrieved(r, stage) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r["id"],
            content=r["content"],
            doc_filename=r.get("filename", ""),
            vector_score=round(r["cosine_score"], 4) if "cosine_score" in r else None,
            bm25_score=round(r["bm25_score"], 4) if "bm25_score" in r else None,
            rrf_score=round(r["rrf_score"], 6) if "rrf_score" in r else None,
            graph_boosted=r.get("graph_boosted", False),
        )

    debug = PipelineDebug(
        vector_results=[_to_retrieved(r, "vector") for r in vec_results],
        bm25_results=[_to_retrieved(r, "bm25") for r in bm25_results],
        rrf_merged=[_to_retrieved(r, "rrf") for r in merged],
        graph_facts=[GraphFact(**f) for f in graph_facts],
        token_usage=total_tokens,
        timing=timing,
    )

    return ChatResponse(
        answer=answer_result["text"],
        sources=[_to_retrieved(r, "final") for r in top_chunks],
        debug=debug,
    )


def _save_chat(user_id: str, question: str, answer: str):
    """Save user question and assistant answer to chat history."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (%s, 'user', %s)",
                (user_id, question),
            )
            cur.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (%s, 'assistant', %s)",
                (user_id, answer),
            )
        conn.commit()

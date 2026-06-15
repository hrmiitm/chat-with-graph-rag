"""
GraphRAG retrieval pipeline:
1. Embed the question (sentence-transformers on CPU)
2. Vector search (pgvector cosine similarity → find relevant chunks)
3. Extract entities from question (LLM)
4. Graph search (Apache AGE — traverse knowledge graph for related facts)
5. Build context (merge chunks + graph facts)
6. Generate answer (LLM with full context)

Every step's results, scores, and timing are returned for frontend debugging.
"""

import logging
import time

from fastapi import APIRouter

from app.config import settings
from app.database import execute_sql, execute_cypher, get_conn
from app.llm_client import embed_single, generate, parse_json_from_llm
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
# Graph Search (Apache AGE)
# =============================================================================

def extract_question_entities(question: str) -> list[str]:
    """Use LLM to extract searchable entity names from the question."""
    prompt = QUESTION_ENTITY_PROMPT.format(question=question)
    result = generate(prompt, max_tokens=512)
    try:
        entities = parse_json_from_llm(result["text"])
        if isinstance(entities, list):
            logger.info(f"Extracted question entities: {entities}")
            return entities
        return []
    except Exception:
        # Fallback: split question into significant words
        stop = {"what", "how", "why", "when", "where", "who", "is", "are", "the", "a", "an",
                "in", "on", "of", "for", "to", "and", "or", "do", "does", "can", "will"}
        words = [w for w in question.lower().split() if w not in stop and len(w) > 2]
        logger.info(f"Fallback question entities: {words}")
        return words


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
# Chat Endpoint
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    GraphRAG retrieval + answer generation.
    Pipeline: embed → vector search → graph search → generate answer.
    Returns answer with full pipeline debug info.
    """
    timing = {}
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    logger.info(f"═══ CHAT QUERY: '{req.question}' (user={req.user_id}) ═══")

    # Step 1: Embed the question
    t0 = time.time()
    q_embedding = embed_single(req.question)
    timing["embed_query_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 1/4] Query embedded in {timing['embed_query_s']}s")

    # Step 2: Vector search — find relevant chunks
    t0 = time.time()
    vec_results = vector_search(q_embedding, settings.top_k)
    timing["vector_search_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 2/4] Vector search: {len(vec_results)} results in {timing['vector_search_s']}s")

    # Step 3: Graph search — extract entities from question, then traverse graph
    t0 = time.time()
    q_entities = extract_question_entities(req.question)
    graph_facts = graph_search(q_entities)
    timing["graph_search_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 3/4] Graph search: {len(graph_facts)} facts for entities {q_entities} "
                f"in {timing['graph_search_s']}s")

    # Graph boosting — mark chunks that contain graph-connected entities
    graph_entity_names = set()
    for f in graph_facts:
        graph_entity_names.add(f["source"].lower())
        graph_entity_names.add(f["target"].lower())

    for item in vec_results:
        content_lower = item["content"].lower()
        item["graph_boosted"] = any(e in content_lower for e in graph_entity_names)

    # Sort: graph-boosted chunks first, then by cosine score
    vec_results.sort(key=lambda x: (x.get("graph_boosted", False), x.get("cosine_score", 0)), reverse=True)

    # Take top K for context
    top_chunks = vec_results[: settings.top_k]

    # Step 4: Build context and generate answer
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

    logger.info(f"[Step 4/4] Answer generated in {timing['generation_s']}s "
                f"({total_tokens['total']} tokens)")

    # Save chat history
    _save_chat(req.user_id, req.question, answer_result["text"])

    # Build response with full debug info
    def _to_retrieved(r) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r["id"],
            content=r["content"],
            doc_filename=r.get("filename", ""),
            vector_score=round(r["cosine_score"], 4) if "cosine_score" in r else None,
            graph_boosted=r.get("graph_boosted", False),
        )

    debug = PipelineDebug(
        vector_results=[_to_retrieved(r) for r in vec_results],
        graph_facts=[GraphFact(**f) for f in graph_facts],
        question_entities=q_entities,
        token_usage=total_tokens,
        timing=timing,
    )

    logger.info(f"═══ CHAT COMPLETE ═══")
    return ChatResponse(
        answer=answer_result["text"],
        sources=[_to_retrieved(r) for r in top_chunks],
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

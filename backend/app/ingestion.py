"""
Document ingestion pipeline:
1. Upload file → extract text (PDF/DOCX/TXT)
2. Split into chunks (sliding window)
3. Generate embeddings (sentence-transformers on CPU)
4. Extract entities & relations (LLM via vLLM)
5. Store chunks + embeddings in pgvector
6. Store entities + relations in Apache AGE graph

Returns full debug info: chunks, entities, relations, token usage, timing.
"""

import io
import json
import logging
import time

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.database import execute_sql_returning, execute_sql, execute_cypher, get_conn
from app.llm_client import embed, generate, parse_json_from_llm
from app.prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_PROMPT
from app.models import IngestResponse, ChunkInfo, EntityInfo, RelationInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingestion"])


# =============================================================================
# Text Extraction
# =============================================================================

def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from PDF, DOCX, or TXT file."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    elif ext == "docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == "txt":
        return content.decode("utf-8", errors="ignore")

    else:
        raise HTTPException(400, f"Unsupported file type: .{ext}. Use PDF, DOCX, or TXT.")


# =============================================================================
# Chunking (sliding window)
# =============================================================================

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of configured size."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# =============================================================================
# Entity & Relation Extraction via LLM
# =============================================================================

def extract_entities_relations(chunks: list[str]) -> tuple[list[dict], list[dict], dict]:
    """
    Use LLM to extract entities and relations from each chunk.
    Returns: (entities, relations, token_usage)
    """
    all_entities = []
    all_relations = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    for i, chunk in enumerate(chunks):
        logger.info(f"[Chunk {i}/{len(chunks)-1}] Extracting entities...")
        prompt = ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk)
        result = generate(prompt, system=ENTITY_EXTRACTION_SYSTEM, max_tokens=2048)

        total_tokens["prompt_tokens"] += result["prompt_tokens"]
        total_tokens["completion_tokens"] += result["completion_tokens"]

        logger.info(f"[Chunk {i}] LLM response ({result['latency_s']}s, "
                     f"{result['completion_tokens']} tokens):\n{result['text'][:500]}")

        # Parse LLM JSON response
        try:
            data = parse_json_from_llm(result["text"])

            chunk_entities = []
            for ent in data.get("entities", []):
                chunk_entities.append({
                    "name": ent["name"],
                    "entity_type": ent.get("type", "Other"),
                    "source_chunk": i,
                })

            chunk_relations = []
            for rel in data.get("relations", []):
                chunk_relations.append({
                    "source": rel["source"],
                    "relation": rel["relation"],
                    "target": rel["target"],
                    "source_chunk": i,
                })

            all_entities.extend(chunk_entities)
            all_relations.extend(chunk_relations)
            logger.info(f"[Chunk {i}] ✓ {len(chunk_entities)} entities, {len(chunk_relations)} relations")

        except Exception as e:
            logger.warning(f"[Chunk {i}] ✗ Failed to parse entities: {e}")
            continue

    total_tokens["total"] = total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]
    return all_entities, all_relations, total_tokens


# =============================================================================
# Store in Apache AGE Graph
# =============================================================================

def store_graph(entities: list[dict], relations: list[dict]):
    """Store extracted entities and relations in the Apache AGE knowledge graph."""
    created_nodes = 0
    created_edges = 0

    # Create entity nodes (MERGE to avoid duplicates)
    for ent in entities:
        name = ent["name"].replace("'", "\\'")
        etype = ent["entity_type"].replace("'", "\\'")
        cypher = f"MERGE (n:Entity {{name: '{name}', type: '{etype}'}})"
        try:
            execute_cypher(cypher, columns="v agtype")
            created_nodes += 1
        except Exception as e:
            logger.warning(f"Failed to create entity node '{name}': {e}")

    # Create relation edges
    for rel in relations:
        src = rel["source"].replace("'", "\\'")
        tgt = rel["target"].replace("'", "\\'")
        rtype = rel["relation"].replace("'", "\\'")
        cypher = (
            f"MATCH (a:Entity {{name: '{src}'}}), (b:Entity {{name: '{tgt}'}}) "
            f"CREATE (a)-[:RELATES_TO {{type: '{rtype}'}}]->(b)"
        )
        try:
            execute_cypher(cypher, columns="e agtype")
            created_edges += 1
        except Exception as e:
            logger.warning(f"Failed to create relation {src}->{tgt}: {e}")

    logger.info(f"Graph stored: {created_nodes} nodes, {created_edges} edges")


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document. Full pipeline:
    extract text → chunk → embed → extract entities → store in DB + graph.
    """
    timing = {}
    logger.info(f"═══ INGESTION START: {file.filename} ═══")

    # 0. Check for duplicate
    existing = execute_sql(
        "SELECT id FROM documents WHERE filename = %s", (file.filename,)
    )
    if existing:
        logger.info(f"Document '{file.filename}' already exists (id={existing[0]['id']}), will re-ingest")

    # 1. Extract text
    t0 = time.time()
    content = await file.read()
    text = extract_text(file.filename, content)
    if not text.strip():
        raise HTTPException(400, "No text could be extracted from the file.")
    timing["extraction_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 1/6] Text extracted: {len(text)} chars in {timing['extraction_s']}s")

    # 2. Store document
    doc = execute_sql_returning(
        "INSERT INTO documents (filename, content) VALUES (%s, %s) RETURNING id, filename",
        (file.filename, text),
    )
    doc_id = doc["id"]
    logger.info(f"[Step 2/6] Document stored: id={doc_id}")

    # 3. Chunk text
    t0 = time.time()
    chunks = chunk_text(text)
    timing["chunking_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 3/6] Chunked: {len(chunks)} chunks in {timing['chunking_s']}s")

    # 4. Generate embeddings
    t0 = time.time()
    embed_result = embed(chunks)
    embeddings = embed_result["embeddings"]
    timing["embedding_s"] = embed_result["latency_s"]
    logger.info(f"[Step 4/6] Embeddings generated: {len(embeddings)} vectors in {timing['embedding_s']}s")

    # 5. Store chunks + embeddings in pgvector
    t0 = time.time()
    chunk_infos = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute(
                    "INSERT INTO chunks (doc_id, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (doc_id, i, chunk, str(emb)),
                )
                chunk_infos.append(ChunkInfo(
                    chunk_index=i,
                    content=chunk,
                    embedding_preview=emb[:5],  # first 5 dims for display
                ))
        conn.commit()
    timing["db_store_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 5/6] Chunks stored in pgvector in {timing['db_store_s']}s")

    # 6. Extract entities & relations via LLM
    t0 = time.time()
    entities, relations, token_usage = extract_entities_relations(chunks)
    timing["graph_extraction_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 6a/6] Extracted {len(entities)} entities, {len(relations)} relations "
                f"in {timing['graph_extraction_s']}s")

    # 7. Store in Apache AGE graph
    t0 = time.time()
    store_graph(entities, relations)
    timing["graph_store_s"] = round(time.time() - t0, 3)
    logger.info(f"[Step 6b/6] Graph stored in {timing['graph_store_s']}s")

    logger.info(f"═══ INGESTION COMPLETE: {file.filename} ═══")
    logger.info(f"  Chunks: {len(chunks)}, Entities: {len(entities)}, "
                f"Relations: {len(relations)}, Tokens: {token_usage.get('total', 0)}")

    return IngestResponse(
        document_id=doc_id,
        filename=file.filename,
        total_chunks=len(chunks),
        chunks=chunk_infos,
        entities=[EntityInfo(**e) for e in entities],
        relations=[RelationInfo(**r) for r in relations],
        token_usage=token_usage,
        timing=timing,
    )

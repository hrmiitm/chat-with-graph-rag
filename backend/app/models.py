"""
Pydantic models for API requests and responses.
Includes pipeline debug models that expose every step to the frontend.
"""

from pydantic import BaseModel
from typing import Optional


# =============================================================================
# Ingestion
# =============================================================================

class ChunkInfo(BaseModel):
    """Single chunk with its metadata."""
    chunk_index: int
    content: str
    embedding_preview: list[float]  # first 5 dims for display


class EntityInfo(BaseModel):
    """Extracted entity from text."""
    name: str
    entity_type: str
    source_chunk: int


class RelationInfo(BaseModel):
    """Extracted relation (triple) from text."""
    source: str
    relation: str
    target: str
    source_chunk: int


class IngestResponse(BaseModel):
    """Full response from document ingestion — shows everything that happened."""
    document_id: int
    filename: str
    total_chunks: int
    chunks: list[ChunkInfo]
    entities: list[EntityInfo]
    relations: list[RelationInfo]
    token_usage: dict         # {"prompt_tokens": N, "completion_tokens": N, "total": N}
    timing: dict              # {"extraction_s": N, "chunking_s": N, "embedding_s": N, ...}


# =============================================================================
# Retrieval / Chat
# =============================================================================

class ChatRequest(BaseModel):
    question: str
    user_id: str = "user_a"


class RetrievedChunk(BaseModel):
    """A chunk retrieved by vector search, with scores."""
    chunk_id: int
    content: str
    doc_filename: str
    vector_score: Optional[float] = None     # cosine similarity
    graph_boosted: bool = False              # whether graph context boosted this chunk


class GraphFact(BaseModel):
    """A fact retrieved from the knowledge graph."""
    source: str
    relation: str
    target: str


class PipelineDebug(BaseModel):
    """Complete debug info for the GraphRAG retrieval pipeline."""
    # Search results
    vector_results: list[RetrievedChunk]
    graph_facts: list[GraphFact]

    # Entities extracted from the question (used for graph search)
    question_entities: list[str] = []

    # Token usage for this query
    token_usage: dict

    # Timing breakdown (seconds)
    timing: dict


class ChatResponse(BaseModel):
    """Full chat response with answer + debug pipeline."""
    answer: str
    sources: list[RetrievedChunk]    # final ranked sources used for answer
    debug: PipelineDebug

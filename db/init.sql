-- GraphRAG Database Schema
-- Extensions: pgvector (vector search), Apache AGE (knowledge graph)
-- This script runs automatically on first container start via docker-entrypoint-initdb.d

-- =============================================================================
-- 1. Enable Extensions
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector for embedding storage + similarity search
CREATE EXTENSION IF NOT EXISTS age;          -- Apache AGE for Cypher graph queries

-- Load AGE and set search path so Cypher functions are accessible
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- =============================================================================
-- 2. Documents Table — stores uploaded files
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    content     TEXT NOT NULL,              -- full extracted text
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- 3. Chunks Table — text chunks with embeddings + full-text search
-- =============================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,           -- position within document
    content     TEXT NOT NULL,              -- chunk text
    embedding   vector(1024),              -- Qwen3-Embedding-0.6B outputs 1024 dims
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for fast full-text (BM25-style) search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- =============================================================================
-- 4. Chat History Table — per-user conversation tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history (user_id, created_at);

-- =============================================================================
-- 5. Knowledge Graph — Apache AGE graph for entities and relations
-- =============================================================================
SELECT create_graph('document_graph');

-- =============================================================================
-- 6. Make AGE search path persistent for all connections
-- =============================================================================
ALTER DATABASE graphrag SET search_path = ag_catalog, "$user", public;

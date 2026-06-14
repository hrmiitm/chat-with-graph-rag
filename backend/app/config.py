"""
Application configuration — reads all settings from environment variables.
Single source of truth for every configurable value.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Database ---
    database_url: str = "postgresql://graphrag:graphrag123@localhost:5432/graphrag"

    # --- vLLM (chat/generation) ---
    vllm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-4B"

    # --- Embedding (runs on CPU via sentence-transformers) ---
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # --- Chunking ---
    chunk_size: int = 512       # characters per chunk
    chunk_overlap: int = 50     # overlap between consecutive chunks

    # --- Retrieval ---
    top_k: int = 10             # candidates from each search method
    rrf_k: int = 60             # RRF constant (standard default)

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton instance — import this everywhere
settings = Settings()

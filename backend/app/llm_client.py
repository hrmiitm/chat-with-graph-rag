"""
LLM and Embedding clients.
- Chat/generation: OpenAI-compatible client pointing to local vLLM
- Embedding: sentence-transformers on CPU (saves GPU for LLM)

Both track and return token usage for frontend debugging.
"""

import json
import logging
import re
import time
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Think-tag stripping (Qwen3 reasoning models wrap output in <think>...</think>)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# vLLM client for chat/generation (GPU)
# ---------------------------------------------------------------------------
_llm_client: OpenAI | None = None


def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key="not-needed",  # vLLM doesn't require auth
        )
    return _llm_client


def generate(prompt: str, system: str = "", max_tokens: int = 1024) -> dict:
    """
    Generate text using the LLM via vLLM's OpenAI-compatible API.
    Automatically strips <think> tags from output.

    Returns:
        {"text": str, "prompt_tokens": int, "completion_tokens": int, "latency_s": float}
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    try:
        resp = _get_llm_client().chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        latency = time.time() - start
        raw_text = resp.choices[0].message.content.strip()
        clean_text = strip_think_tags(raw_text)
        return {
            "text": clean_text,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "latency_s": round(latency, 3),
        }
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return {"text": f"[LLM Error: {e}]", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0}


def check_llm_health() -> bool:
    """Return True if vLLM is reachable and serving the model."""
    try:
        models = _get_llm_client().models.list()
        return len(models.data) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Embedding model (CPU via sentence-transformers)
# ---------------------------------------------------------------------------
_embed_model = None


def _get_embed_model():
    """Lazy-load the embedding model on first use."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model} (CPU)")
        start = time.time()
        _embed_model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
            trust_remote_code=True,
        )
        logger.info(f"Embedding model loaded in {time.time() - start:.1f}s")
    return _embed_model


def warmup_embed_model():
    """Force-load the embedding model so first real request isn't slow."""
    _get_embed_model()
    logger.info("Embedding model warmed up")


def embed(texts: list[str]) -> dict:
    """
    Generate embeddings for a list of texts.

    Returns:
        {"embeddings": list[list[float]], "latency_s": float}
    """
    model = _get_embed_model()
    start = time.time()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    latency = time.time() - start
    return {
        "embeddings": vectors.tolist(),
        "latency_s": round(latency, 3),
    }


def embed_single(text: str) -> list[float]:
    """Convenience: embed a single text and return the vector."""
    result = embed([text])
    return result["embeddings"][0]


def parse_json_from_llm(text: str) -> any:
    """Robustly parse JSON output from LLM, stripping thinking tags and wrapper text."""
    # 1. Strip thinking tags (already done by generate(), but just in case)
    text = strip_think_tags(text)

    # 2. Extract content between first '{' and last '}' or first '[' and last ']'
    first_dict = text.find("{")
    first_list = text.find("[")

    if first_dict != -1 and (first_list == -1 or first_dict < first_list):
        start_idx = first_dict
        end_idx = text.rfind("}")
    elif first_list != -1:
        start_idx = first_list
        end_idx = text.rfind("]")
    else:
        return json.loads(text.strip())

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx : end_idx + 1].strip()
        return json.loads(json_str)

    return json.loads(text.strip())

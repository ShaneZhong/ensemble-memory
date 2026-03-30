#!/usr/bin/env python3
"""embeddings.py — ONNX/sentence-transformers embedding service for ensemble memory.

Provides lazy-loaded embedding generation using all-MiniLM-L6-v2 (384-dim, ~80MB).
All functions degrade gracefully if sentence-transformers is not installed.

Env vars:
    ENSEMBLE_MEMORY_EMBED_MODEL   Model name (default: all-MiniLM-L6-v2)
"""

import logging
import math
import os
import sys
import warnings
from typing import Optional

logger = logging.getLogger("ensemble_memory.embeddings")

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("ENSEMBLE_MEMORY_EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Availability check ────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer as _ST
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.info(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )

# ── Lazy model cache ──────────────────────────────────────────────────────────

_model: Optional[object] = None


def _get_model() -> Optional[object]:
    """Load and cache the model on first use."""
    global _model
    if not _AVAILABLE:
        return None
    if _model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = _ST(MODEL_NAME, device="cpu")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> Optional[list[float]]:
    """Generate embedding for a single text. Returns 384-dim vector or None."""
    model = _get_model()
    if model is None:
        return None
    vec = model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def get_embeddings(texts: list[str]) -> Optional[list[list[float]]]:
    """Generate embeddings for multiple texts (batched). Returns list of 384-dim vectors or None."""
    if not texts:
        return []
    model = _get_model()
    if model is None:
        return None
    vecs = model.encode(texts, convert_to_numpy=True, batch_size=32)
    return [v.tolist() for v in vecs]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(
    query_embedding: list[float],
    candidates: list[dict],
    threshold: float = 0.7,
    top_k: int = 5,
) -> list[dict]:
    """Find top-k candidates above threshold.

    Each candidate dict must have an 'embedding' key (list[float]).
    Returns candidates sorted by similarity desc, with 'similarity' score added.
    """
    results = []
    for candidate in candidates:
        emb = candidate.get("embedding")
        if emb is None:
            continue
        sim = cosine_similarity(query_embedding, emb)
        if sim >= threshold:
            results.append({**candidate, "similarity": sim})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# ── Setup helper ──────────────────────────────────────────────────────────────

def setup_embeddings() -> None:
    """Install sentence-transformers if not already available."""
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "sentence-transformers"]
    )
    logger.info("sentence-transformers installed. Restart your process to use it.")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not _AVAILABLE:
        logger.error("Run setup_embeddings() or: pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading model: %s", MODEL_NAME)
    vec = get_embedding("hello world")
    logger.info("Embedding dim: %d", len(vec))
    logger.info("First 8 values: %s", [round(v, 4) for v in vec[:8]])

    sim = cosine_similarity(vec, vec)
    logger.info("Self-similarity: %.6f", sim)

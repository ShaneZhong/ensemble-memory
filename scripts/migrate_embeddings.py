#!/usr/bin/env python3
"""migrate_embeddings.py — Re-embed all memories with the current embedding model.

Usage:
    python scripts/migrate_embeddings.py

This script re-embeds all active memories using the configured embedding model.
Run after upgrading the embedding model (e.g., MiniLM 384-dim → BGE-M3 1024-dim).

Idempotent — safe to re-run. Will overwrite existing embeddings with new ones.
"""

import logging
import sys
from pathlib import Path

# Make hooks importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_HOOKS_DIR = _PROJECT_ROOT / "hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    import db
    import embeddings

    logger.info("Embedding model: %s (dim=%d)", embeddings.MODEL_NAME, embeddings.EMBEDDING_DIM)

    # Check model availability
    test_vec = embeddings.get_embedding("test")
    if test_vec is None:
        logger.error("Embedding model not available. Install sentence-transformers and ensure model is downloaded.")
        sys.exit(1)

    logger.info("Model produces %d-dim vectors", len(test_vec))

    count = db.reembed_all_memories()
    logger.info("Done. Re-embedded %d memories.", count)


if __name__ == "__main__":
    main()

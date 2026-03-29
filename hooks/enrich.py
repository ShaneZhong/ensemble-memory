#!/usr/bin/env python3
"""Contextual enrichment for ensemble memory system.

Enriches memory content with contextual prefixes before embedding, improving
retrieval quality. Two paths:
  - KG path (free): uses entity neighborhood from knowledge graph
  - LLM path (~5-10s): uses Ollama qwen2.5:3b with type-specific prompts

Enrichment is best-effort and non-fatal. If it fails, the raw memory content
is preserved with its original embedding.
"""

import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ensemble_memory.enrich")

_HOOKS_DIR = Path(__file__).parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("ENSEMBLE_MEMORY_MODEL", "qwen2.5:3b")
ENRICHMENT_ENABLED = os.environ.get("ENSEMBLE_MEMORY_ENRICH_ENABLED", "1") == "1"
MIN_ENRICHMENT_IMPORTANCE = int(os.environ.get("ENSEMBLE_MEMORY_MIN_ENRICH_IMPORTANCE", "6"))
MAX_ENRICHED_WORDS = 80
MIN_ENRICHED_WORDS = 5
OLLAMA_TIMEOUT = 15  # seconds

# Stop words for novelty checking
_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "in", "it", "to", "and", "or", "for",
    "of", "on", "at", "be", "do", "i", "you", "we", "this", "that",
    "with", "from", "not", "are", "was", "has", "have", "will", "can",
    "use", "as", "by", "my", "me", "if", "so", "but", "its", "also",
])

# Type-specific LLM prompt templates
_LLM_TEMPLATES = {
    "correction": (
        "Add brief context to this coding correction. "
        "State what technology/tool it involves, what was wrong, and the correct approach. "
        "Keep it under 80 words. Do NOT start with 'I'.\n\n"
        "Correction: {content}\n\nEnriched version:"
    ),
    "procedural": (
        "Add brief context to this coding rule. "
        "State what technology it applies to, when to use it, and why it matters. "
        "Keep it under 80 words. Do NOT start with 'I'.\n\n"
        "Rule: {content}\n\nEnriched version:"
    ),
    "semantic": (
        "Add brief context to this technical fact. "
        "State what system or project it relates to and why it's important. "
        "Keep it under 80 words. Do NOT start with 'I'.\n\n"
        "Fact: {content}\n\nEnriched version:"
    ),
}


def enrich_memory(
    content: str,
    memory_type: str,
    importance: int,
    entity_names: list,
    subject: Optional[str],
) -> Optional[dict]:
    """Enrich a memory with contextual prefix to improve retrieval quality.

    Tries KG path first (free, uses entity neighborhood), then falls back to
    LLM path (uses Ollama). Returns dict with 'text' and 'quality' keys, or
    None if enrichment is skipped or fails validation.
    """
    if not ENRICHMENT_ENABLED:
        return None
    if memory_type == "episodic":
        return None
    if importance < MIN_ENRICHMENT_IMPORTANCE:
        return None

    # Try KG path first
    enriched_text = _enrich_via_kg(content, memory_type, entity_names, subject)
    used_kg = enriched_text is not None

    # Fall back to LLM path
    if enriched_text is None:
        enriched_text = _enrich_via_llm(content, memory_type, subject)

    if enriched_text is None:
        return None

    if not _validate_enrichment(enriched_text, content):
        return None

    quality = _compute_quality(enriched_text, content, used_kg)
    return {"text": enriched_text, "quality": quality}


def _enrich_via_kg(
    content: str,
    memory_type: str,
    entity_names: list,
    subject: Optional[str],
) -> Optional[str]:
    """Enrich memory using entity neighborhood from the knowledge graph.

    Returns enriched text string, or None if KG path is not applicable.
    """
    if not entity_names or len(entity_names) < 2:
        return None

    try:
        import kg
        import db as _db
        # Cold-start guard: skip KG path if entity graph is too sparse
        _cs_conn = _db.get_db()
        entity_count = _cs_conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        _cs_conn.close()
        if entity_count < 50:
            return None  # Too few entities for meaningful context

        neighborhood = kg.kg_entity_neighborhood(entity_names[:5], max_depth=1, max_neighbors=3)
        formatted_prefix = neighborhood.get("formatted_prefix", "")
        if not formatted_prefix or len(formatted_prefix.strip()) < 10:
            return None

        first_entity = subject or entity_names[0]
        enriched = (
            f"{memory_type.capitalize()} about {first_entity}: "
            f"{content} Context: {formatted_prefix[:200]}"
        )
        words = enriched.split()
        if len(words) > MAX_ENRICHED_WORDS:
            enriched = " ".join(words[:MAX_ENRICHED_WORDS])
        return enriched
    except Exception:
        return None


def _enrich_via_llm(
    content: str,
    memory_type: str,
    subject: Optional[str],
) -> Optional[str]:
    """Enrich memory using Ollama LLM with type-specific prompts.

    Returns enriched text string, or None if LLM call fails or type unsupported.
    """
    if memory_type not in _LLM_TEMPLATES:
        return None

    prompt = _LLM_TEMPLATES[memory_type].format(content=content)

    try:
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 256,
                "temperature": 0.3,
            },
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            result = json.loads(resp.read())

        text = result.get("response", "").strip()
        words = text.split()
        if len(words) > MAX_ENRICHED_WORDS:
            text = " ".join(words[:MAX_ENRICHED_WORDS])
        return text if text else None

    except Exception as exc:
        logger.warning("[enrich] LLM path failed: %s", exc)
        return None


def _validate_enrichment(enriched: str, original: str) -> bool:
    """Validate enriched text meets quality requirements.

    Checks: word count bounds, no first-person start, and novelty vs original.
    """
    words = enriched.split()

    if len(words) < MIN_ENRICHED_WORDS or len(words) > MAX_ENRICHED_WORDS:
        return False

    if enriched.startswith("I "):
        return False

    original_words = set(original.lower().split())
    enriched_words = [w.lower() for w in words]
    novel_words = [
        w for w in enriched_words
        if w not in original_words and w not in _STOP_WORDS
    ]
    if len(novel_words) < 2:
        return False

    return True


def _compute_quality(enriched: str, original: str, used_kg: bool) -> float:
    """Compute a quality score in [0, 1] for enriched text.

    Combines length score, novelty score, and a KG bonus.
    """
    words = enriched.lower().split()
    length_score = min(len(words) / MAX_ENRICHED_WORDS, 1.0)

    original_words = set(original.lower().split())
    novel_words = [w for w in words if w not in original_words and w not in _STOP_WORDS]
    novelty_score = min(len(novel_words) / 5.0, 1.0)

    kg_bonus = 0.2 if used_kg else 0.0

    return min((length_score * 0.4 + novelty_score * 0.4 + kg_bonus + 0.2), 1.0)


def enrich_batch(min_importance: int = 6, limit: int = 100, dry_run: bool = False) -> dict:
    """Enrich all un-enriched memories above the importance threshold.

    Processes memories in batches, applying KG or LLM enrichment. Rate-limits
    LLM calls to 1 per second. Returns counts of processed/enriched/skipped.
    """
    import db

    conn = db.get_db()
    processed = 0
    enriched_count = 0
    skipped_count = 0

    try:
        rows = conn.execute(
            """
            SELECT id, content, memory_type, importance, subject, session_id
            FROM memories
            WHERE enriched_text IS NULL
              AND importance >= ?
              AND superseded_by IS NULL
            LIMIT ?
            """,
            (min_importance, limit),
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        processed += 1
        mem_id = row[0]
        content = row[1]
        memory_type = row[2]
        importance = row[3]
        subject = row[4]
        session_id = row[5]

        # Load entity names from KG for this memory
        entity_names = []
        try:
            import kg
            kg_conn = db.get_db()
            entity_rows = kg_conn.execute(
                """
                SELECT DISTINCT e.name FROM kg_entities e
                JOIN kg_appears_in ai ON ai.entity_id = e.id
                JOIN kg_episodes ep ON ep.id = ai.episode_id
                WHERE ep.session_id = ?
                LIMIT 10
                """,
                (session_id,),
            ).fetchall()
            entity_names = [r["name"] for r in entity_rows]
            kg_conn.close()
        except Exception:
            pass  # KG lookup is best-effort

        result = enrich_memory(content, memory_type, importance, entity_names, subject)
        if result and not dry_run:
            db.store_enrichment(mem_id, result["text"], result["quality"])
            enriched_count += 1
            # Only rate-limit LLM calls, not free KG enrichments
            if result.get("quality", 0) < 0.8:  # KG path scores higher due to bonus
                time.sleep(1.0)
        else:
            skipped_count += 1

    return {
        "processed": processed,
        "enriched": enriched_count,
        "skipped": skipped_count,
    }

#!/usr/bin/env python3
"""evolution.py — A-MEM relationship classification between memories.

Classifies relationships between a new memory and existing similar memories
using Ollama. Called asynchronously by the daemon background job, NOT inline
in the stop hook.

Link types (matching kg_memory_links.link_type CHECK constraint):
  RELATED, CONTRADICTS, SUPERSEDES, EVOLVED_FROM, SUPPORTS, REFINES, ENABLES, CAUSED_BY
"""

import json
import logging
import os
import time
import urllib.request
from typing import Optional

logger = logging.getLogger("ensemble_memory.evolution")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 15  # seconds

_VALID_LINK_TYPES = frozenset([
    "RELATED", "CONTRADICTS", "SUPERSEDES", "EVOLVED_FROM",
    "SUPPORTS", "REFINES", "ENABLES", "CAUSED_BY",
])

_CLASSIFICATION_PROMPT = """\
You are a memory relationship classifier. Given a NEW memory and a list of EXISTING memories, classify the relationship between the new memory and each existing memory.

NEW MEMORY:
{new_content}

EXISTING MEMORIES:
{existing_list}

For each existing memory, output the relationship type. Valid types:
- SUPPORTS: new memory reinforces or agrees with existing
- REFINES: new memory adds detail or nuance to existing
- CONTRADICTS: new memory conflicts with existing
- SUPERSEDES: new memory replaces existing (newer version of same info)
- EVOLVED_FROM: new memory builds on existing (causal evolution)
- ENABLES: new memory makes existing possible or easier
- CAUSED_BY: new memory was caused by existing
- RELATED: general connection (use when no stronger type fits)

Output ONLY valid JSON:
{{"relationships": [{{"existing_id": "ID", "link_type": "TYPE", "strength": 0.0-1.0}}]}}

RULES:
- Only classify relationships with confidence. If unsure, use RELATED with low strength.
- Strength 0.8+ means strong evidence. 0.5 means moderate. Below 0.3, skip it.
- Output empty list if no meaningful relationships exist.
"""

# V2 prompt with few-shot examples for improved classification accuracy
_CLASSIFICATION_PROMPT_V2 = """\
You are a memory relationship classifier. Given a NEW memory and EXISTING memories, classify how they relate.

RELATIONSHIP TYPES:
- SUPPORTS: new reinforces existing ("Always use pytest" supports "Use pytest fixtures for setup")
- REFINES: new adds detail to existing ("Use pytest with -x flag" refines "Use pytest for tests")
- CONTRADICTS: new conflicts with existing ("Use unittest" contradicts "Never use unittest")
- SUPERSEDES: new replaces existing ("Use SQLite not PostgreSQL" supersedes "Use PostgreSQL")
- EVOLVED_FROM: new builds on existing ("Add cross-encoder reranking" evolved from "Implement RRF fusion")
- ENABLES: new makes existing possible ("Install sentence-transformers" enables "Use cross-encoder")
- CAUSED_BY: new was caused by existing ("FK constraint error" caused by "Wrong table referenced")
- RELATED: general connection, no stronger type fits

EXAMPLES:
New: "Always use ruff for linting, never flake8"
Existing: [ID: abc] "Use flake8 for Python linting"
Output: {{"relationships": [{{"existing_id": "abc", "link_type": "SUPERSEDES", "strength": 0.9}}]}}

New: "pytest fixtures should use function scope by default"
Existing: [ID: def] "Always use pytest for testing"
Output: {{"relationships": [{{"existing_id": "def", "link_type": "REFINES", "strength": 0.7}}]}}

New: "The build passed all 323 tests"
Existing: [ID: ghi] "Use SQLite for the database"
Output: {{"relationships": []}}

NOW CLASSIFY:
NEW MEMORY:
{new_content}

EXISTING MEMORIES:
{existing_list}

Output ONLY valid JSON:
{{"relationships": [{{"existing_id": "ID", "link_type": "TYPE", "strength": 0.0-1.0}}]}}

STRENGTH CALIBRATION:
- 0.8-1.0: Obvious, clear relationship (e.g., direct contradiction or replacement)
- 0.5-0.7: Moderate relationship (e.g., one refines the other)
- 0.3-0.5: Weak but real connection (e.g., related topic)
- Below 0.3: Skip — not worth recording
"""


def classify_relationships(
    new_memory_id: str,
    new_content: str,
    similar_memories: list[dict],
) -> list[dict]:
    """Classify relationships between a new memory and similar existing ones.

    Args:
        new_memory_id: ID of the new memory
        new_content: Content text of the new memory
        similar_memories: List of dicts with keys: id, content

    Returns:
        List of dicts: {source_id, target_id, link_type, strength}
    """
    if not similar_memories:
        return []

    # Build existing memories list for prompt
    existing_lines = []
    valid_ids = set()
    for i, mem in enumerate(similar_memories[:5], 1):
        mem_id = mem.get("id", "")
        content = mem.get("content", "")
        existing_lines.append(f"{i}. [ID: {mem_id}] {content}")
        valid_ids.add(mem_id)

    use_v2 = os.environ.get("ENSEMBLE_MEMORY_AMEM_PROMPT_V2", "1") == "1"
    template = _CLASSIFICATION_PROMPT_V2 if use_v2 else _CLASSIFICATION_PROMPT
    prompt = template.format(
        new_content=new_content,
        existing_list="\n".join(existing_lines),
    )

    # Call Ollama
    response_text = _call_ollama(prompt)
    if not response_text:
        return []

    # Parse response
    return _parse_classification(new_memory_id, response_text, valid_ids)


def _call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama and return the response text, or None on failure."""
    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except Exception as exc:
        logger.warning("[evolution] Ollama call failed: %s", exc)
        return None


def _parse_classification(
    new_memory_id: str,
    response_text: str,
    valid_ids: set[str],
) -> list[dict]:
    """Parse Ollama's JSON response into a list of relationship dicts."""
    # Extract JSON from response (may have markdown fences)
    text = response_text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("[evolution] Could not parse classification response")
                return []
        else:
            return []

    relationships = data.get("relationships", [])
    result = []

    for rel in relationships:
        existing_id = rel.get("existing_id", "")
        link_type = rel.get("link_type", "").upper()
        strength = float(rel.get("strength", 0.5))

        # Validate
        if existing_id not in valid_ids:
            continue
        if link_type not in _VALID_LINK_TYPES:
            link_type = "RELATED"
        if strength < 0.3:
            continue  # Too weak to record

        result.append({
            "source_entity_id": new_memory_id,
            "target_entity_id": existing_id,
            "link_type": link_type,
            "strength": min(1.0, max(0.0, strength)),
        })

    return result

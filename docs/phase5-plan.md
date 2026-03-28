# Phase 5: Contextual Enrichment — Implementation Plan

## Goal
Improve retrieval quality by enriching memory content with contextual prefixes BEFORE embedding. Raw memories like "Use 4 spaces" become "Python coding preference: Use 4 spaces for indentation, related to PEP 8 compliance" — making embeddings more specific and retrievable.

Design doc target: 35-67% retrieval quality improvement.

## Architecture (adapted to our stack)

We use SQLite + all-MiniLM-L6-v2 (not Milvus + BGE-M3 as originally designed).

```
[Stop hook extracts memory]
    |
    v
[store_memory.py] ──> [db.insert_memory()] ──> SQLite
    |
    v
[enrich.py] ──────────────────────────────────────────
    |                                                  |
    v                                                  v
[KG path: kg_entity_neighborhood()]        [LLM path: Ollama qwen2.5:3b]
(2+ entities → graph prefix, free)         (entity-sparse → type-specific prompt)
    |                                                  |
    v                                                  v
[Validate: 5-150 words, novelty check]
    |
    v
[Store enriched_text in memories.enriched_text column]
    |
    v
[Re-embed enriched_text via daemon /embed]
[Store new embedding (replaces raw embedding)]
```

## Components

### 1. New file: hooks/enrich.py (~200 lines)

**enrich_memory(content, memory_type, importance, entities, subject)** → enriched_text or None

Two enrichment paths:

**Path A: KG-based (zero LLM cost)**
- Condition: 2+ entities extracted in this turn AND kg_entities count >= 200
- Process: Call `kg.kg_entity_neighborhood(entity_names, max_depth=1)` → get relationships
- Generate prefix: "{memory_type} about {subject}: {content}. Context: {entity} is a {type}, related to {relationships}"
- Cost: ~5ms (SQLite query only)

**Path B: LLM-based (Ollama call)**
- Condition: <2 entities OR kg cold-start (< 200 entities)
- Process: Type-specific prompt to qwen2.5:3b
- Templates:
  - correction: "Summarize this coding correction with context about what was wrong and the correct approach: {content}"
  - procedural: "Add context to this coding rule — when it applies, why it matters: {content}"
  - semantic: "Add context to this technical fact — what system/project it relates to: {content}"
  - episodic: Skip enrichment (episodic memories are already contextual)
- Cost: ~5-10s (Ollama inference)

**Validation:**
- Min 5 words, max 150 words
- Must not start with "I" (first-person)
- Novelty check: enriched text must contain >= 2 words not in original content
- If validation fails, return None (use raw content)

**Quality score:**
- enrichment_quality = (length_ok + novelty_ok + has_entity_context) / 3.0
- Stored alongside enriched_text

### 2. Schema migration: memories.enriched_text + memories.enrichment_quality

Add two columns to memories table:
```sql
ALTER TABLE memories ADD COLUMN enriched_text TEXT;
ALTER TABLE memories ADD COLUMN enrichment_quality REAL DEFAULT 0.0;
```

Migration via `ensure_enrichment_columns()` pattern (like `ensure_embedding_column()`).

### 3. Integration in store_memory.py

After embedding generation (line ~84-100), add enrichment call:
```python
# Enrich memory (async-safe, non-blocking)
if importance >= 6:
    enriched = enrich.enrich_memory(content, mem_type, importance, entities, subject)
    if enriched:
        db.store_enrichment(mem_id, enriched["text"], enriched["quality"])
        # Re-embed with enriched text
        embed_result = daemon_embed(enriched["text"])
        if embed_result:
            db.store_embedding(mem_id, embed_result)
```

### 4. Daemon search uses enriched embeddings

No daemon code changes needed — embeddings are already stored per-memory. The re-embed step replaces the raw embedding with the enriched one. Cosine search automatically uses the better embedding.

### 5. Re-enrichment on supersession

When a memory supersedes another, the new memory's enrichment should incorporate context from the old memory. Add to `store_memory.py` after supersession detection:
```python
if superseded_old_id:
    old_mem = db.get_memory(superseded_old_id)
    if old_mem and old_mem.get("enriched_text"):
        # Re-enrich with old context
        enriched = enrich.enrich_with_predecessor(content, old_mem["enriched_text"])
```

### 6. Batch re-enrichment command

`hooks/enrich_batch.py` — CLI tool to re-enrich existing memories:
```bash
python3 hooks/enrich_batch.py --min-importance 6 --limit 100
```
Processes memories that have embeddings but no enriched_text.

## What's NOT included (deferred)
- Milvus integration (we use SQLite + daemon)
- BGE-M3 re-embedding (we use all-MiniLM-L6-v2)
- Batch processing of 15+ chunks per prompt (overkill for per-turn processing)
- enrichment_quality in Milvus schema (no Milvus)

## Estimated scope
- New files: 1 (enrich.py ~200 lines)
- Modified files: 3 (db.py migration, store_memory.py integration, extraction prompt)
- New tests: ~15-20
- Total new LOC: ~350-450
- RAM impact: 0 MB (reuses existing Ollama + daemon)
- Write path latency: +5ms (KG path) or +5-10s (LLM path, async)

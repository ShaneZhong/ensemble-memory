# Phase 7: Recall Quality & A-MEM Evolution

**Status**: COMPLETE
**Date**: 2026-03-31
**Prerequisite**: Phase 6 (lifecycle management) — complete
**Target**: Close the largest remaining gaps between implementation and spec

---

## Motivation

Phase 1-6 built a solid **collection** pipeline. But the system's value is measured by **recall quality** — does the right memory surface at the right time? Currently:

1. **A-MEM evolution (Step 10)** is unimplemented — inter-memory relationships don't exist
2. **SessionEnd safety net** is missing — memories only captured by per-turn regex triage
3. **Extraction prompt** lacks `trigger_condition` — reinforcement matching is less precise
4. **Recall scoring** uses simple cosine+BM25+temporal — no KG context boost or cross-encoder reranking
5. **UserPromptSubmit recall** only queries the daemon `/search` — no decision vault or KG graph context

These are the 5 highest-impact gaps. Closing them completes the core ensemble architecture.

---

## Gap Analysis: Spec vs Implementation

```
SPEC STEP                        STATUS          IMPACT
─────────────────────────────────────────────────────────
Step 1-2: Triage regex           ✅ Done         —
Step 3: Ollama extraction        ✅ Done         —
Step 4: Routing + importance     ⚠️ Partial      trigger_condition missing from prompt
Step 5: Temporal register        ✅ Done         —
Step 6: Embedding pass           ✅ Done         (384-dim, spec says 1024 BGE-M3)
Step 7: Contextual enrichment    ✅ Done         —
Step 8: Markdown log             ✅ Done         —
Step 9a: Semantic write          ✅ Done         (SQLite, spec says Milvus)
Step 9b: KG write                ✅ Done         —
Step 9c: ClawMem write           ✅ Done         —
Step 10: A-MEM evolution         ❌ Missing      No inter-memory relationships
Step 11: Reinforcement           ✅ Done         —
SessionEnd safety net            ❌ Missing      Memories lost to regex blind spots
UserPromptSubmit recall          ⚠️ Basic        No decision vault, no KG context
Read flow: composite scoring     ⚠️ Basic        No validity gates, no KG boost
Read flow: cross-encoder rerank  ❌ Missing      Spec requires for Stop hook
```

### What we're NOT doing in Phase 7 (deferred to Phase 8+)

- BGE-M3 1024-dim migration (requires re-embedding all memories)
- Milvus Lite migration (SQLite + sentence-transformers works fine at current scale)
- Cross-encoder reranking — spec says "ALWAYS use for Stop hook retrieval" but requires running ms-marco-MiniLM-L-6-v2 (~85MB) on a separate llama-server port. Deferred because: (a) additional ~85MB RAM on M4, (b) requires deploying and managing a second model server, (c) current RRF fusion is adequate at <500 memories. Will revisit when memory count exceeds 1K.
- SessionStart remaining gaps: temporal validity gates on loaded content, semantic recent context injection (top 30 lines, importance >= 5). These improve quality but are not blocking.
- Cross-project memory federation
- Memory visualization/dashboard
- PreCompact hook integration

---

## Architecture: Phase 7 Pipeline Changes

```
CURRENT PIPELINE (Stop hook):
  triage → extract → store_memory → [embed, enrich, supersede, KG, decisions] → done

PHASE 7 PIPELINE (Stop hook):
  triage → extract → store_memory → [embed, enrich, supersede, KG, decisions]
                                         ↓
                                   queue A-MEM job ──→ daemon background
                                                        ↓
                                                   [search top-5 similar]
                                                   [classify relationships]
                                                   [write kg_memory_links]

PHASE 7 ADDITION (SessionEnd hook):
  session_end.py → full-transcript extraction → dedup by content_hash → store_memory

PHASE 7 RECALL UPGRADE (UserPromptSubmit):
  daemon /search → RRF(cosine + BM25 + temporal)
                 + decision vault BM25 query
                 + KG entity neighborhood context
                 → composite score → top-5 returned
```

---

## Sprint Plan (6 sprints)

### Sprint 1: Extraction Prompt Upgrade (~45 LOC)

**Goal**: Add `trigger_condition` to extraction prompt so reinforcement matching has structured data.

**Background**: Phase 6 Sprint 1a-prereq identified this as needed but the extraction prompt was not updated during Phase 6 implementation. Sprint 1 completes this deferred work. The current `store_memory.py` uses `mem.get("rule", "")` — the LLM does extract `rule` but never `trigger_condition`, making reinforcement less precise.

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `hooks/prompts/extraction.txt` | Add `trigger_condition`, `anti_pattern`, `correct_pattern` fields | ~10 |
| `hooks/store_memory.py` | Use `trigger_condition` for reinforcement match_text (fallback to `rule` then `content`) | ~5 |
| `tests/test_phase7.py` | Test trigger_condition extraction and reinforcement matching | ~30 |

**Changes**:
```
extraction.txt:
  Add to memory schema:
    "trigger_condition": "when this rule applies (e.g., 'when writing Python tests')",
    "anti_pattern": "what NOT to do (e.g., 'import unittest')",
    "correct_pattern": "what TO do (e.g., 'import pytest')"

store_memory.py:
  Change: match_text = mem.get("trigger_condition", "") or mem.get("rule", "") or content
```

**Acceptance**:
- [x] Extraction prompt includes trigger_condition field
- [x] Reinforcement uses trigger_condition as primary match text
- [x] Backward compatible — existing memories without trigger_condition still work

---

### Sprint 2: SessionEnd Safety Net — SQLite Write (~100 LOC)

**Goal**: Close the gap where `session_end.sh` catches missed memories but only writes them to markdown (via `write_log.py`), not to SQLite. This means safety-net memories are invisible to recall.

**Current state**: `session_end.sh` already does:
- Parse transcript from hook payload ✅
- Build full-session text, truncated to 4000 chars ✅
- Call `extract.py` with empty signals (no triage gate) ✅
- Dedup against today's markdown log via substring match ✅
- Write new memories to markdown via `write_log.py` ✅
- Write to SQLite via `store_memory.py` ❌ **THIS IS THE GAP**

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `hooks/session_end.sh` | Add `store_memory.py` call for SQLite write after `write_log.py` | ~5 |
| `hooks/session_end.py` (NEW) | Python wrapper for SQLite dedup + store (called from .sh) | ~60 |
| `tests/test_phase7.py` | SessionEnd tests including negative cases | ~50 |

**How it works**:

```
session_end.sh (AUGMENTED, not replaced):
  ... existing extraction + markdown dedup + write_log.py ...
  # NEW: also write to SQLite for recall
  python3 "$HOOKS_DIR/session_end.py" "$FILTERED_EXTRACTION" "$SESSION_ID"

session_end.py:
  1. Receive filtered extraction JSON and session_id as args
  2. Parse extraction, get memories list
  3. For each memory:
     a. Compute content_hash = SHA-256(content)
     b. Check if content_hash exists in memories table (any session)
     c. If not duplicate: call store_memory._store_to_sqlite([mem], session_id)
  4. Log: "SessionEnd SQLite: N new, M already in DB"
  5. On any Ollama/DB failure: skip gracefully (markdown already written by .sh)

Note: Ollama cold-model risk — if the session was short and no Stop hook fired,
Ollama may not be warm. The existing session_end.sh already handles the Ollama
call and its timeout. session_end.py only handles the SQLite write for the
already-extracted memories. No additional Ollama call needed.
```

**Negative/edge case tests**:
- Empty transcript → no crash, no memories stored
- All memories already in DB → 0 new, dedup works
- Corrupted extraction JSON → graceful skip
- DB locked by daemon → retry or skip (non-fatal)

**Acceptance**:
- [x] Safety-net memories written to SQLite (not just markdown)
- [x] Dedup by content_hash against existing DB memories
- [x] `session_end.sh` augmented (not replaced) — existing logic preserved
- [x] No additional Ollama call (reuses extraction from .sh)
- [x] Graceful failure — markdown is already written, SQLite is best-effort
- [x] Negative/edge case tests pass

---

### Sprint 3: A-MEM Evolution (~300 LOC)

**Goal**: Implement Step 10 — classify relationships between new and existing memories.

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `hooks/evolution.py` (NEW) | A-MEM relationship classification via daemon + Ollama | ~150 |
| `hooks/db_lifecycle.py` | Add `insert_memory_link()`, `get_memory_links()` | ~40 |
| `daemon/embedding_daemon.py` | Add A-MEM background job to timer chain | ~30 |
| `hooks/store_memory.py` | Queue memory ID for async evolution after store | ~15 |
| `tests/test_phase7.py` | A-MEM tests (unit + queue lifecycle) | ~100 |

**How it works**:

```
store_memory.py:
  After successful insert:
    INSERT INTO kg_sync_state ('amem_queue_<mem_id>', mem_id, now)

daemon background job (runs with other bg jobs every 6h, wrapped in own try/except):
  1. SELECT key, value, updated_at FROM kg_sync_state
     WHERE key LIKE 'amem_queue_%'
  2. For each queued item:
     - value = memory_id (the memory to evolve)
     - updated_at = creation timestamp (for 7-day expiry check)
     - If (now - updated_at) > 7 days: DELETE and skip (expired)
     a. Call /search to get top-5 similar memories
     b. Build Ollama prompt: "Classify the relationship between these memories"
     c. Parse response → list of (source_id, target_id, link_type, strength)
     d. Write to kg_memory_links
     e. DELETE queue entry from kg_sync_state
  3. On Ollama failure: leave entry in queue (updated_at unchanged).
     Next run retries. After 7 days, auto-discarded by expiry check.

Note: kg_sync_state is reused as a simple queue (key=job_id, value=payload,
updated_at=enqueue_time). The spec defines memory_pipeline_queue for this
purpose but we defer that table — kg_sync_state suffices at current scale.

SQLite contention: Both the stop hook (store_memory.py) and daemon A-MEM job
write to the same DB. WAL mode + BEGIN IMMEDIATE + 100ms retry backoff (existing
pattern from Phase 6 background jobs) handles this. The A-MEM job is added to
the serialized timer chain so it never runs concurrently with other background
jobs (community detection, GC, etc.).

A-MEM classifications are logged to the daemon's standard logger at INFO level,
including memory_id, link_type, and strength for each classification. This
enables manual review via the existing debug log at
$ENSEMBLE_MEMORY_DIR/logs/daemon_debug.log.

evolution.py:
  classify_relationships(memory_id, similar_memories) -> list[MemoryLink]
  - Uses Ollama qwen2.5:3b
  - Link types: SUPPORTS, REFINES, CONTRADICTS, ENABLES, CAUSED_BY, RELATED, EVOLVED_FROM, SUPERSEDES
  - Returns structured list for db insert
```

**Acceptance**:
- [x] New memories are queued for A-MEM evolution
- [x] Daemon processes queue asynchronously (not in stop hook)
- [x] Relationships written to kg_memory_links table
- [x] Ollama timeout handled gracefully (retry via queue persistence)
- [x] 7-day auto-discard for stale queue entries
- [x] A-MEM job wrapped in own try/except (won't break timer chain)
- [x] No regression in stop hook latency
- [x] Queue lifecycle tests: enqueue, dequeue, retry, 7-day expiry
- [x] Classifications logged at INFO level for manual review

---

### Sprint 4: Recall Quality — Validity Gates + Session Start Decisions (~100 LOC)

**Goal**: Add validity gates to recall and inject decisions at session start.

**What ALREADY EXISTS** (no changes needed):
- Decision vault BM25 query — `_bm25_search()` already queries `decisions_fts` + `memories` keyword search (daemon lines 282-387)
- KG entity neighborhood context — `_get_kg_context()` already does FTS5 entity search + 2-hop BFS + community-aware sorting (daemon lines 215-279)
- KG context appended to /search output — already done at daemon line 490-492

**What's NEW in this sprint**:
1. **Validity gates in cosine ranking** — currently `_search()` filters by `temporal_score < MIN_TEMPORAL_SCORE` but does NOT filter superseded, gc_eligible, or expired memories. These slip through.
2. **Session start decision injection** — `session_start.py` loads standing rules but doesn't include recent decisions from the vault.
3. **Source annotations** — hits don't indicate whether they came from cosine, BM25/decisions, or KG.

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `daemon/embedding_daemon.py` | Add validity gate checks in `_search()` cosine loop | ~25 |
| `hooks/session_start.py` | Query decision vault top-3 by project at session start | ~35 |
| `tests/test_phase7.py` | Validity gate + session start decision tests | ~40 |

**Changes**:

```
daemon _search() cosine loop — ADD:
  if mem.get("superseded_by") is not None:
      continue
  if mem.get("gc_eligible", 0) == 1:
      continue
  valid_to = mem.get("valid_to")
  if valid_to and valid_to < now:
      continue

session_start.py — ADD after standing rules:
  decisions = db.search_decisions_bm25(query="", project=project, limit=3)
  if decisions:
      lines.append("## Recent Decisions")
      for d in decisions:
          lines.append(f"- [{d['decision_type']}] {d['content'][:200]}")
```

**Acceptance**:
- [x] Superseded/gc_eligible/expired memories excluded from /search results
- [x] Session start includes top-3 recent decisions from vault
- [x] Source annotations on hits (cosine, bm25, decisions_fts)
- [x] No regression in /search latency (gates are O(1) per memory)
- [x] E2E test: insert superseded memory → verify it's excluded from recall

---

### Sprint 5: Recall Scoring Upgrade (~80 LOC)

**Goal**: Replace the inline scoring formula with a clean composite scoring function.

**Existing formula** (daemon line 473):
```python
final = rrf_score * (0.4 + t_score * 0.3 + importance * 0.3)
```
This is a multiplicative blend where RRF is the base and temporal+importance are multipliers (range 0.4-1.0). Issues:
- `confidence` (retrieval confidence, reduced on contradiction) is never applied
- Validity gates are handled elsewhere (Sprint 4 adds them to the cosine loop) but not in the scoring function itself
- The formula is inline, not testable independently

**New formula** (extracted to a named function for testability):

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `daemon/embedding_daemon.py` | Extract `composite_score()` function, apply confidence | ~50 |
| `tests/test_phase7.py` | Scoring tests including feature flag toggle | ~30 |

```python
def composite_score(
    rrf_score: float,           # Normalized RRF from fusion
    temporal_score: float,      # From db.temporal_score()
    importance: int,            # 1-10
    confidence: float = 1.0,    # Retrieval confidence (reduced on contradiction)
) -> float:
    # Weight profile (default, can be tuned per query type)
    w_semantic = 0.5
    w_temporal = 0.3
    w_importance = 0.2

    importance_score = importance / 10.0

    final = (
        w_semantic * rrf_score
        + w_temporal * temporal_score
        + w_importance * importance_score
    )
    final *= confidence  # Key improvement: contradicted memories rank lower

    return final
```

**Migration**: The new formula is an additive weighted sum (vs. the old multiplicative blend). This changes ranking behavior — a memory with high importance but low semantic match can now rank higher. This is intentional and matches the spec (Section 6.2). The feature flag `ENSEMBLE_MEMORY_COMPOSITE_SCORING=0` falls back to the old formula for safe rollback.

**Acceptance**:
- [x] `composite_score()` is a standalone, testable function
- [x] `confidence` field now affects ranking (contradicted memories rank lower)
- [x] Feature flag `ENSEMBLE_MEMORY_COMPOSITE_SCORING=0` restores old formula
- [x] Test: old formula behavior preserved when flag is off
- [x] Test: confidence=0.5 produces lower score than confidence=1.0 for same inputs

---

### Sprint 6: Integration Testing + Hardening (~150 LOC)

**Goal**: End-to-end tests covering the full Phase 7 pipeline.

**Files**:
| File | Change | Lines |
|------|--------|-------|
| `tests/test_phase7.py` | E2E integration tests | ~100 |
| Various | Bug fixes from testing | ~50 |

**Test scenarios**:

1. **E2E A-MEM evolution**: Insert 3 related memories → verify kg_memory_links created with correct link_types
2. **E2E SessionEnd safety net**: Insert memories via stop hook, then call session_end.py with same + new content → verify only new content written to SQLite
3. **E2E recall quality**: Insert correction + decision + KG entities → call `_search()` → verify all three sources appear in results
4. **E2E reinforcement with trigger_condition**: Insert procedural with trigger_condition → insert same rule with different wording → verify reinforcement_count incremented
5. **E2E composite scoring**: Insert superseded + active memories with same content relevance → verify active memory ranks higher
6. **E2E validity gates**: Insert expired (valid_to in past) + gc_eligible + superseded memories → verify all excluded from search results
7. **Regression**: All existing tests still pass (currently 277)

**Acceptance criteria (measurable)**:
- [x] All 7 E2E test scenarios pass
- [x] Zero regressions in existing test suite
- [x] A-MEM queue lifecycle verified (enqueue → process → delete; retry on failure; 7-day expiry)
- [x] Total test count: existing 277 + Phase 7 unit tests + Phase 7 E2E tests all green
- [x] No new warnings or errors in test output

---

## Sprint Ordering Rationale

Sprints are ordered by dependency + risk:
1. **Extraction prompt** — smallest change, broadest downstream impact (enables better reinforcement)
2. **SessionEnd safety net** — lower risk than A-MEM (no Ollama call in new code), higher immediate value (catches missed memories NOW). No dependency on A-MEM.
3. **A-MEM evolution** — highest complexity sprint. Depends on Sprint 1 (better extraction data). Independent of Sprint 2.
4. **Validity gates + session start decisions** — depends on existing data; benefits from Sprint 2/3 having more memories in DB
5. **Composite scoring** — depends on Sprint 4 (validity gates move to cosine loop, scoring formula must account for them)
6. **Integration testing** — depends on all previous sprints

## Estimated Scope

| Sprint | New LOC | Test LOC | Files Modified | Files Created |
|--------|---------|----------|---------------|---------------|
| 1. Extraction prompt | ~15 | ~30 | 2 | 0 |
| 2. SessionEnd safety net | ~65 | ~50 | 1 | 1 |
| 3. A-MEM evolution | ~300 | ~100 | 3 | 1 |
| 4. Validity gates + decisions | ~60 | ~40 | 2 | 0 |
| 5. Composite scoring | ~50 | ~30 | 1 | 0 |
| 6. Integration testing | ~50 | ~100 | various | 0 |
| **Total** | **~540** | **~350** | — | **2 new files** |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| A-MEM Ollama latency in daemon | Async background job, not in stop hook. 60s timer. Retry queue with 7-day discard. |
| A-MEM classification quality with qwen2.5:3b | 3b model may produce low-quality relationship classifications. Log all classifications for manual review. Accuracy calibration deferred to Phase 8 (target: >70%). Sprint 6 E2E tests validate on known examples. |
| Ollama cold model at SessionEnd | If session was short and no Stop hook fired, Ollama may not be warm. Mitigated: `session_end.sh` already handles the Ollama call and timeout. `session_end.py` only does the SQLite write — no additional Ollama call. If extraction fails in .sh, .py is never invoked. |
| SessionEnd transcript too large for Ollama | Existing `session_end.sh` truncates to last 4000 chars. No change needed. |
| KG context bloats recall output | Already capped: `_get_kg_context()` limits to 5 entity names, 2-hop depth, 3 neighbors. No change needed. |
| Composite scoring changes recall behavior | Feature-flagged: `ENSEMBLE_MEMORY_COMPOSITE_SCORING=1`. Default on. Set to 0 to restore old formula. |
| Existing tests break | Run full suite after each sprint. Review after each sprint (threshold ≥7/10). |
| SQLite contention (daemon A-MEM + stop hook) | WAL mode + BEGIN IMMEDIATE + 100ms retry backoff. A-MEM job serialized in timer chain (never concurrent with other background jobs). |

## Dependencies

- **Ollama qwen2.5:3b** — already installed and used by extraction
- **sentence-transformers** — already installed for daemon
- **NetworkX** — already installed for community detection
- No new dependencies required

## NOT in Scope (Phase 8+)

- BGE-M3 1024-dim re-embedding (requires model swap + full re-index)
- Milvus Lite migration
- Cross-encoder reranking — spec says "ALWAYS" for Stop hook retrieval, but requires ms-marco-MiniLM (~85MB) on separate llama-server port. Deferred: current RRF is adequate at <500 memories, additional RAM pressure on M4.
- SessionStart: temporal validity gates on loaded content, semantic recent context injection (top 30 lines, importance >= 5). Improves quality but not blocking.
- PreCompact hook integration
- Cross-project memory federation
- Memory visualization/dashboard
- A-MEM accuracy calibration (target: >70%) — Sprint 6 validates on known examples, formal calibration is Phase 8
- `memory_pipeline_queue` table (spec has it, `kg_sync_state` queue suffices at current scale)
- `clawmem_sessions`, `observations`, `co_activation_events` tables (ClawMem session management)
- `decisions_vec` virtual table (requires sqlite-vec extension)

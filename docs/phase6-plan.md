# Phase 6: Full Ensemble + Evolution — Implementation Plan

## Goal
Transform separate expert systems into a unified ensemble with emergent memory lifecycle management. Phase 6 closes the loop: memories don't just get stored and retrieved — they evolve, get reinforced, contradict each other, form communities, decay, get promoted, and get garbage collected.

## Current State (What Already Exists)

| Component | Schema | Logic | Status |
|-----------|--------|-------|--------|
| `reinforcement_count` column | YES | Partial (detection only, no increment) | 50% |
| `promotion_candidate` column | YES | Logging only, no CLAUDE.md write | 10% |
| `temporal_score` column | YES | Computed on-the-fly in daemon, not cached | 30% |
| `supersession_events` table | YES | Written by detect_supersession/detect_content_supersession | 70% |
| `supersession_depth_limits` table | YES | Config exists, no chain traversal code | 20% |
| `kg_decay_config` table | YES | Config exists, no enforcement code | 20% |
| `community_id` on kg_entities | YES | Column exists, no algorithm | 5% |
| Decision vault (Phase 4) | YES | Full CRUD + BM25 search | 100% |
| Contextual enrichment (Phase 5) | YES | KG + LLM paths, validation, batch | 100% |

## Architecture

```
[Stop hook captures memory]
    |
    v
[store_memory.py] ─── existing Phase 1-5 pipeline ───> SQLite + Markdown
    |
    v
[Reinforcement check] ── procedural trigger_condition match?
    |                           |
   no                         yes
    |                           |
    v                           v
[normal insert]          [increment_reinforcement()]
                               |
                          [queue promotion to daemon if count >= 5]

[Daemon background jobs (serialized threading.Timer chain)]
    |
    ├── [Promotion pipeline] ── check criteria ──> write CLAUDE.md (fcntl lock)
    ├── [Supersession event bus] ── trilateral sync ──> temporal + KG + contextual
    ├── [Chain depth pruning] ── walk chains ──> GC old memories per type limits
    ├── [Community detection] ── Louvain on KG ──> kg_entities.community_id
    ├── [Relationship decay] ── check kg_decay_config ──> expire stale edges
    ├── [Temporal score caching] ── batch compute ──> memories.temporal_score
    └── [Garbage collection] ── chain-pruned OR forgotten+superseded ──> gc_eligible=1

[DEFERRED Phase 6b: A-MEM Evolution]
    ├── [evolve_memory()] ── daemon background job ──> kg_memory_links
    └── [causal_link_propagation()] ──> decisions.causal_*_edges
```

## Sprint Plan

### Sprint 1: Reinforcement + Promotion Pipeline (Core Lifecycle)
**Estimated: ~250 LOC across db.py, store_memory.py, new promote.py**

**1a-prereq. Verify `trigger_condition` in extraction prompt**
- Reinforcement matching at store_memory.py:71 checks `mem.get("trigger_condition", "")`. This field must be present in the LLM extraction output.
- Verify `hooks/prompts/extraction.txt` includes `trigger_condition` in the JSON schema. If missing, reinforcement is dead code — add it to the prompt.
- Also fix existing `print()` at store_memory.py:76-80 → `logger.info()` while touching this code.

**1a. Reinforcement increment function** (`db.py`)
- `increment_reinforcement(memory_id)` → bump `reinforcement_count`, update `stability`
- Stability rules:
  - count=2: stability += 0.1
  - count=3: stability += 0.2, set promotion_candidate=1
  - count>=5: stability = 1.0 (permanent)
- Impact on decay: `lambda_eff = decay_rate * (1 - stability * 0.8)`

**1b. Fix reinforcement flow in store_memory.py**
- Currently: when reinforcement detected (count > 0), skips insert entirely (line 81 `continue`)
- The `continue` also skips entity upserts, embedding, enrichment, and supersession for reinforced memories. This is correct — a reinforced memory doesn't need re-processing, only a count bump.
- Refactor `get_reinforcement_count()` → `get_reinforcement_match()` returning `(count, memory_id)` instead of just `int`
- **Critical: escape SQL wildcards** (`%`, `_`) in trigger_condition before LIKE query. Use `ESCAPE '\\'` clause to prevent false positive matches.
- Before `continue`: call `increment_reinforcement(existing_id)` to bump count + stability
- After increment: call `check_and_promote(existing_id)` if count >= PROMOTION_THRESHOLD
- **Critical: wrap reinforcement+promotion calls in try/except** (non-fatal, like enrichment at store_memory.py:109-142). If increment_reinforcement raises, remaining memories in the batch must still process.
- This is a ~30-40 LOC change to `_store_to_sqlite()`, not a trivial few-line fix

**1c. Promotion pipeline** (new file: `hooks/promote.py`, ~100 lines)
- `check_and_promote(memory_id)` → reads memory, checks criteria:
  - reinforcement_count >= 5
  - last accessed within 180 days
  - memory_type == "procedural"
- **Runs as daemon background job** (not inline in stop hook) to avoid adding latency to timing-sensitive stop hook path. Queued via daemon's threading.Timer chain.
- Writes to project's CLAUDE.md under `## Learned Behaviors` section
- Appends: `- {content} (importance: {importance}, reinforced: {count}x)`
- Creates the section if it doesn't exist
- Idempotent: checks if content already in CLAUDE.md before writing
- File locking: `fcntl.flock()` advisory lock, 5s timeout. All CLAUDE.md writers must cooperate (add code comment documenting this contract). Skip if lock unavailable — next session will retry.
- Configurable target: CLAUDE.md path from `ENSEMBLE_MEMORY_CLAUDE_MD` env var, defaults to `{project_root}/CLAUDE.md`

**1d. Tests** (~20 tests)
- Reinforcement increment at counts 1,2,3,4,5 → correct stability values (count=4 is between thresholds: no new flag, stability unchanged from count=3)
- Stability impact on decay formula (verify lambda_eff at each stability level)
- Promotion criteria (count >= 5, within 180 days, procedural only)
- Promotion rejected: count=4 (below threshold), count=5 but stale (> 180 days), non-procedural type
- CLAUDE.md write format and idempotency (duplicate promotion is no-op)
- CLAUDE.md edge cases: file doesn't exist (create it), section doesn't exist (append section), section already has content (append to section), file is read-only (skip with warning)
- `get_reinforcement_match()` returns (count, memory_id) correctly
- `get_reinforcement_match()` with multiple matches: picks most recent non-superseded
- `get_reinforcement_match()` with SQL wildcards in trigger_condition (`%`, `_`): escaped correctly, no false positives
- `increment_reinforcement()` on non-existent memory_id: no-op or raises cleanly
- store_memory.py flow: reinforced memory skips insert but increments count
- store_memory.py flow: increment_reinforcement raises → remaining memories still process (try/except)

### Sprint 2: Supersession Event Bus + Chain Pruning (Lifecycle Management)
**Estimated: ~200 LOC across db.py, kg.py, enrich.py**

**2a. Trilateral supersession sync** (`db.py`)
- `process_supersession_events()` → scans unprocessed events
- **Per-event, per-expert processing**: for each event, check each `processed_by_X` flag independently. Only process experts where flag = 0. This ensures partial failures don't reprocess already-completed experts.
- Uses single connection per event (pass conn internally, commit per event, close in finally). Prevents transaction spanning issues while keeping atomicity per-event.
- Temporal processor: already mostly done (superseded_by set at detection time)
  - Just needs to set `processed_by_temporal = 1`
- KG processor: `_process_supersession_kg(old_id, new_id)`
  - Find relationships involving old memory's entities
  - Set `valid_until` on contradicted edges (where subject/object match old memory's entities)
  - Handle edge case: old memory has no entities → no-op, set flag anyway
  - Set `processed_by_kg = 1`
- Contextual processor: `_process_supersession_contextual(old_id, new_id)`
  - Clear `enriched_text` on old memory (mark stale). If already NULL, no-op.
  - Set `processed_by_contextual = 1`

**2b. Chain depth pruning** (`db.py`)
- `get_supersession_chain(memory_id)` → walk chain via `superseded_by` pointers
- `enforce_chain_depth_limits()` → for each active memory:
  - Walk chain, check depth vs `supersession_depth_limits` for that type
  - If chain too long: mark oldest links as `gc_eligible = 1`
- Type limits (from existing config):
  - procedural: max 3, correction: max 2, semantic: max 3, episodic: max 5

**2c. Tests** (~15 tests)
- Event bus processes all three experts (temporal, KG, contextual)
- Idempotency: reprocessing doesn't duplicate work (re-run sets same flags)
- Partial processing: if KG fails, temporal and contextual still process
- Chain depth correctly computed (depth 1, 2, 3 chains)
- Pruning marks correct memories as gc_eligible at each type's limit
- Protected memories (importance >= 9) never GC'd even if chain exceeds limit
- Empty supersession_events table: no-op, no errors
- Chain with missing intermediate memory (data corruption): graceful handling

### Sprint 3: Community Detection + Relationship Decay (KG Evolution)
**Estimated: ~200 LOC across kg.py, daemon/embedding_daemon.py**

**3a. Community detection** (`kg.py`)
- `detect_communities()` → uses NetworkX Louvain on entity-relationship graph
- Build graph: entities as nodes, relationships as weighted edges (confidence as weight)
- Run `community.louvain_communities()` or `community.greedy_modularity_communities()`
- Write results to `kg_entities.community_id`
- Fallback: if NetworkX unavailable, use simple connected-components via SQLite CTE

**3b. Relationship decay** (`kg.py`)
- `apply_relationship_decay()` → check each relationship against `kg_decay_config`
- For non-permanent predicates: if `created_at + decay_window_days < now`:
  - Reduce confidence by 50% (soft decay, not hard delete)
  - If confidence drops below 0.1: mark as expired
- Permanent predicates (NULL decay window): skip
- Idempotency: only decay once per day (use `kg_sync_state` timestamp)

**3c. Community-aware retrieval** (`daemon/embedding_daemon.py`)
- In `_get_kg_context()`: when building neighborhood, prefer entities in same community
- Minor enhancement: group formatted prefix by community for coherence

**3d. Tests** (~16 tests)
- Community detection assigns IDs to connected entities
- Disconnected entities get different community IDs
- Community detection on empty graph: no-op, no errors
- Community detection on single entity: assigns community_id = 0
- All singleton entities (no relationships): each gets unique community
- Relationship decay reduces confidence correctly (50% per window)
- Permanent predicates (NULL decay window) don't decay
- Expired relationships (confidence < 0.1) excluded from neighborhood queries
- Relationship with NULL created_at: skip gracefully (don't crash)
- Decay idempotency: only runs once per day via kg_sync_state
- NetworkX fallback: connected-components CTE assigns groups when NetworkX unavailable
- Community detection with entity cap (> 5000 entities: log warning, skip)
- Community-aware retrieval: same-community entities preferred in _get_kg_context
- Community-aware retrieval: community_id NULL for all entities → falls back to existing behavior (no crash)

### Sprint 4: Temporal Score Caching (Performance)
**Estimated: ~150 LOC across db.py, daemon/embedding_daemon.py**

**4a. Prerequisite: consolidate `_temporal_score()` into single source** (`db.py`)
- Currently duplicated: `db.py` (line ~474) and `embedding_daemon.py` (line ~105) with different signatures
- Consolidate into `db.temporal_score(decay_rate, stability, t_days, access_count, d=0.5)` as the single source
- Daemon imports and calls `db.temporal_score()` instead of its own copy
- This prevents triple-maintenance when Phase 7 calibration tunes the formula

**4b. Temporal score batch computation** (`db.py`)
- `compute_temporal_scores()` → batch update all non-superseded memories
- Uses `db.temporal_score()` per row (consolidated function from 4a)
- Process in chunks of 100 rows (LIMIT clause) to keep write locks short
- Update `score_computed_at` to prevent redundant computation
- Scheduling: daemon runs `compute_temporal_scores()` on startup + via `threading.Timer` every 6 hours

**4c. Daemon uses cached temporal scores** (`daemon/embedding_daemon.py`)
- In `_search()`: use `temporal_score` from DB if `score_computed_at` is recent (< 6h)
- Fallback to on-the-fly computation via `db.temporal_score()` if stale
- Reduces per-query computation cost

**4d. Tests** (~10 tests)
- Consolidated `temporal_score()` matches original daemon formula exactly
- Temporal score batch computation produces correct values
- Score caching prevents redundant computation
- Batch computation on zero memories: no-op
- Batch computation on 100+ memories: completes within 1s
- Stale cache detection (score_computed_at > 6h ago) triggers recompute
- Daemon uses cached score when fresh (mock DB with recent score_computed_at)
- Daemon falls back to on-the-fly when stale (mock DB with old score_computed_at)
- Chunked processing: 200 memories processed in 2 chunks of 100

### Sprint 5: A-MEM Evolution Pass (Full Ensemble) — DEFERRED TO PHASE 6b
**Rationale for deferral (per eng review):**
- Highest-risk sprint: LLM accuracy untested (spec's own Phase 7 calibration target: > 70%)
- Highest-latency: Ollama call per memory (5-15s) in stop hook path approaches 60s timeout
- Best-effort/non-fatal by design — lifecycle management works without it
- Ship lifecycle (Sprints 1-4, 6) first, add evolution as Phase 6b

**When to implement (Phase 6b scope):**
- `hooks/evolution.py` (~200 lines): A-MEM relationship classification via daemon /search + Ollama
- `db.py`: `insert_memory_link()`, `get_memory_links()`, `ensure_causal_columns()` migration
- `store_memory.py`: integration call after enrichment, non-fatal
- Causal link propagation to decisions table
- Evolution retry queue via kg_sync_state
- **Critical prerequisite**: rearchitect as daemon background job (not inline in store_memory.py) to avoid stop hook timeout risk. Queue memory IDs for background evolution.
- ~16 tests covering classification, CRUD, causal propagation, queue, idempotency

**Design deviation documented**: spec says reinforcement (Step 11) comes after A-MEM (Step 10). Reinforcement triggers on `trigger_condition` match, not A-MEM output. No data dependency — independent ordering is correct.

### Sprint 6: Garbage Collection + Integration (System Coherence)
**Estimated: ~150 LOC across db.py, daemon/embedding_daemon.py**

**6a. Garbage collection** (`db.py`)
- `run_garbage_collection()` → soft-delete eligible memories
- Two eligibility paths (OR, not AND — reviewer feedback):
  1. Chain-pruned: `gc_eligible = 1 AND gc_protected = 0`
  2. Naturally forgotten: `temporal_score < 0.005 AND gc_protected = 0 AND superseded_by IS NOT NULL`
- gc_protected: memories with importance >= 9 are protected (never GC'd)
- Soft delete: mark `gc_eligible = 1`, exclude from retrieval
- Does NOT delete rows — preserves history for audit

**6b. GC scheduling**
- Run on daemon startup (after temporal score computation)
- Uses `threading.Timer` in daemon, same mechanism as temporal score batch
- Configurable interval via config.toml (default: 24h)

**6c. Superseded memory filtering in daemon**
- In `_search()`: verify `WHERE superseded_by IS NULL` is already in cosine query
- The existing query at embedding_daemon.py line ~158 already filters this. Sprint 6c is a validation step, not new code.

**6d. End-to-end integration validation**
- Full pipeline test: insert → reinforce → supersede → evolve → decay → GC
- Verify all experts process supersession events
- Verify community IDs are assigned
- Verify all 147+ existing tests still pass (regression gate)

**6e. Tests** (~12 tests)
- GC marks chain-pruned memories
- GC marks naturally forgotten + superseded memories
- Protected memories (importance >= 9) survive GC regardless of score
- Non-superseded low-score memories are NOT GC'd (safety: only GC superseded or chain-pruned)
- Empty database: GC is no-op, no errors
- Full lifecycle test: insert → reinforce 5x → check CLAUDE.md → supersede → check event bus → decay → GC
- Superseded memories excluded from daemon search results

## Acceptance Criteria

| Metric | Target | How to Measure |
|--------|--------|---------------|
| New test count | >= 73 | `python3 tests/test_phase6.py` (20+15+14+10+12 + 2 community retrieval) |
| All tests pass | 100% | New tests + all 147 existing tests (regression gate) |
| Code coverage (new code) | >= 85% | Branch coverage of new functions |
| Reinforcement → promotion | Works end-to-end | Insert procedural 5x → promotion queued to daemon |
| Supersession event bus | 3/3 experts process | Assert all processed_by flags = 1 |
| Chain depth limits | Enforced | Chain exceeding limit → gc_eligible set |
| Community detection | Assigns clusters | Connected entities share community_id |
| Relationship decay | Respects config | Expired relationships lose confidence |
| Temporal score consolidated | Single source | db.temporal_score() used by daemon and batch |
| Temporal score caching | < 6h staleness | Score matches on-the-fly computation |
| GC eligibility | Correct | Two-path: chain-pruned OR forgotten+superseded |
| Daemon search quality | No regressions | Superseded memories excluded from results |
| Background job safety | No lock contention | Serialized timer chain, BEGIN IMMEDIATE, retry |
| No print() in new code | Use logging module | All new code uses logger, not print() |

## New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `hooks/promote.py` | CLAUDE.md promotion pipeline with file locking | ~100 |
| `tests/test_phase6.py` | Phase 6 unit tests (73+ tests) | ~550 |

## Modified Files

| File | Changes | Est. Lines Changed |
|------|---------|-------------------|
| `hooks/db.py` | +reinforcement increment, +chain traversal, +temporal batch (consolidated), +GC, +wildcard escaping | ~200 |
| `hooks/kg.py` | +community detection (with entity cap), +relationship decay (with idempotency) | ~130 |
| `hooks/store_memory.py` | +reinforcement call (try/except wrapped), +promotion queue, refactor get_reinforcement flow, fix print→logger | ~50 |
| `daemon/embedding_daemon.py` | +cached temporal scores (import db.temporal_score), +background job timer chain, +promotion job, +community context | ~100 |
| `hooks/enrich.py` | +stale prefix clearing (supersession contextual processor) | ~15 |

## Deferred to Phase 6b

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `hooks/evolution.py` | A-MEM relationship classification + causal propagation + retry queue | ~200 |
| `hooks/db.py` additions | `insert_memory_link()`, `get_memory_links()`, `ensure_causal_columns()` | ~60 |
| `tests/test_phase6b.py` | A-MEM evolution tests | ~120 |

## Dependencies

- **NetworkX** for community detection (Louvain algorithm)
  - Fallback: connected-components via SQLite recursive CTE if NetworkX unavailable
  - Install: `pip install networkx` (~1.5MB, pure Python)
- **Ollama qwen2.5:3b** for A-MEM evolution (already installed)
- All other dependencies already present

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| A-MEM LLM call adds latency to stop hook | DEFERRED to Phase 6b. Will be rearchitected as daemon background job (not inline in store_memory.py) to avoid 60s timeout risk. |
| NetworkX dependency | Pure-Python fallback via SQLite CTE for connected components |
| Community detection instability across sessions | Expected — community_id is advisory, not deterministic. Retrieval uses it as soft preference, not hard filter. |
| GC accidentally deletes valuable memory | Soft delete only, gc_protected for importance >= 9, two-path eligibility requires superseded OR chain-pruned. |
| Chain pruning breaks supersession history | Pruning marks gc_eligible, doesn't delete rows. Full history preserved. |
| Temporal score drift between cached and on-the-fly | Fallback to on-the-fly if cache > 6h old |
| SQLite write contention (daemon + stop hook + background jobs) | Background jobs (temporal batch, GC, community detection) serialize via single `threading.Timer` chain — only one runs at a time. All use short transactions with `BEGIN IMMEDIATE` to fail fast on lock. Retry with 100ms backoff, max 3 attempts. |
| CLAUDE.md write race condition (concurrent sessions) | Promotion uses atomic write pattern: read → check → write with `fcntl.flock()` advisory lock. Lock timeout 5s, skip promotion if lock unavailable (next session will retry). |
| Ollama cold model for A-MEM evolution | DEFERRED to Phase 6b. Will include retry queue in kg_sync_state with 7-day discard policy. |
| NetworkX Louvain on large graphs (10K+ entities) | Entity count cap: skip community detection if > 5000 entities (log warning). Typical usage: < 500 entities. |
| Community-aware retrieval non-determinism | Sprint 3c is a soft enhancement (prefer same-community entities in neighborhood). If community_id is NULL, falls back to existing behavior. No hard dependency on community stability. |

## Design Deviations from Spec

| Spec Says | Plan Does | Rationale |
|-----------|-----------|-----------|
| Step 11 (reinforcement) comes after Step 10 (A-MEM) | Reinforcement in Sprint 1, A-MEM in Sprint 5 | No data dependency. Reinforcement triggers on `trigger_condition` match, not A-MEM output. Independent ordering is correct. |
| kg_memory_links is "new table" | Table already exists in DDL | `CREATE TABLE IF NOT EXISTS` handles this. Sprint 5a verifies schema matches design. |
| GC eligibility: `gc_eligible = 1 AND temporal_score < 0.005` | Two-path: chain-pruned OR forgotten+superseded | Spec's single-path is too narrow. Non-chain-pruned superseded memories also need GC. |
| Community detection runs at "SessionEnd" | Runs on daemon startup + 6h timer | No SessionEnd hook exists. Daemon startup is the correct trigger point. |
| Stale prefix re-enrichment creates new enrichment | Clears enriched_text, lets next access re-enrich | Simpler. New memory already gets enriched on insert. Old memory's stale prefix just needs clearing. |
| kg_decay_config cold-start threshold: 200 entities | Uses 50 (matching Phase 5) | Consistency with existing cold-start guard in enrich.py. |

## NOT in Scope (Phase 7+)

- Cross-project memory federation
- Multi-user memory isolation
- Real-time streaming evolution (batch only)
- Milvus/vector DB migration
- BGE-M3 re-embedding
- Memory visualization/dashboard UI
- A-MEM accuracy calibration (Phase 7 target: > 70%)

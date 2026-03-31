# Ensemble Memory System — Roadmap

**Last updated**: 2026-04-01
**Spec**: `/Users/shane/Documents/playground/ai_memory/synthesis/final_design.md`

---

## Completed

| Phase | Name | Commit | Tests |
|-------|------|--------|-------|
| 1 | Core capture pipeline (triage → extract → store → embed) | — | 116 |
| 2 | Embedding daemon (cosine + BM25 + RRF search) | — | 19 |
| 3 | Knowledge graph (entities, relationships, FTS5, 2-hop BFS) | — | incl. in P1 |
| 4 | Decision vault (typed decisions, BM25 search, importance decay) | — | incl. in P1 |
| 5 | Contextual enrichment (KG prefix, LLM enrichment, re-embedding) | — | incl. in P1 |
| 6 | Lifecycle (reinforcement, supersession, chain pruning, GC, promotion) | `e84bcd1` | 111 |
| 7 | Recall quality & A-MEM evolution | `ebb7565` | 46 |
| 8 | Recall Quality II — Cross-encoder & Calibration | `ac81c88` | 57 |
| 9 | Embedding Upgrade & Scale (BGE-M3 1024-dim, pipeline queue) | — | 22 |

**Total tests**: 402, all passing (1 pre-existing flaky test in Phase 5).

---

## Phase 9.2 (Deferred): Milvus Lite Migration

**Priority**: LOW (trigger: >1K memories, currently 60)
**Why**: SQLite cosine scan is adequate at current scale. Milvus Lite provides faster vector search but adds a dependency.

| # | Task | LOC est. | Spec ref |
|---|------|----------|----------|
| 9.2 | **Milvus Lite migration** — Replace SQLite cosine scan with Milvus Lite for vector search. Keep SQLite for metadata. Threshold: >1K memories. | ~200 | Section 3.2 |

---

## Phase 10: Hooks & Capture Completeness

**Priority**: MEDIUM
**Why**: Two hook types in the spec remain unimplemented. Capture coverage is incomplete.

| # | Task | LOC est. | Spec ref |
|---|------|----------|----------|
| 10.1 | **PreCompact hook** — Save context summary before Claude Code compresses conversation. Captures memories that would otherwise be lost to context truncation. | ~80 | Section 5.4 |
| 10.2 | **ClawMem session tables** — `clawmem_sessions`, `observations`, `co_activation_events`. Track which memories co-activate in the same retrieval, enabling better future ranking. | ~120 | Section 4.3 |
| 10.3 | **`decisions_vec` virtual table** — Vector search over decisions (requires sqlite-vec extension or Milvus). | ~40 | Section 4.2 |

---

## Phase 11: Monitoring & Operations

**Priority**: LOW (but grows with usage)
**Why**: No visibility into system health. Silent failures go unnoticed.

| # | Task | LOC est. | Spec ref |
|---|------|----------|----------|
| 11.1 | **Health dashboard** — Web UI (simple Flask/FastAPI) showing: memory count, KG stats, daemon uptime, extraction success rate, reinforcement frequency, A-MEM queue depth. | ~300 | — |
| 11.2 | **Memory browser** — Search and browse memories, view KG graph, inspect decay curves, see supersession chains. | ~400 | — |
| 11.3 | **Export/import** — Backup/restore memories to JSON. Migration between machines. | ~100 | — |
| 11.4 | **Alerting** — Log-based alerts for: daemon down, Ollama unreachable, extraction failure rate >20%, queue depth >50. | ~80 | — |

---

## Phase 12: Multi-project & Federation

**Priority**: LOW
**Why**: Currently single-project. Spec envisions cross-project memory sharing.

| # | Task | LOC est. | Spec ref |
|---|------|----------|----------|
| 12.1 | **Cross-project query** — Search memories across all projects, with project-scoped relevance weighting. | ~100 | Section 8.1 |
| 12.2 | **Memory federation** — Share procedural memories (importance >= 8, reinforcement >= 3) across projects automatically. | ~150 | Section 8.2 |
| 12.3 | **Generalize to Gemini CLI** — Abstract hook layer so the same memory system works with Gemini CLI (not just Claude Code). | ~200 | Section 1.2 |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-28 | SQLite over PostgreSQL | Simpler, local, 100% free. Adequate at <10K memories. |
| 2026-03-28 | SQLite KG over Kuzu | Kuzu archived Oct 2025. SQLite adjacency tables suffice. |
| 2026-03-28 | all-MiniLM-L6 over BGE-M3 | Faster, smaller. BGE-M3 deferred to Phase 9. |
| 2026-03-29 | Defer A-MEM to Phase 7 | Highest latency risk, untested accuracy. Lifecycle first. |
| 2026-03-30 | `amem_memory_links` table | Separate from `kg_memory_links` (entity-level FKs). |
| 2026-03-31 | 6h A-MEM retry interval | Matches existing daemon bg job cycle. 60s would be wasteful. |
| 2026-04-01 | BGE-M3 over MiniLM-L6 | Chinese content support needed. 1024-dim captures more info. |
| 2026-04-01 | Defer Milvus Lite to >1K memories | Only 60 active memories. SQLite cosine scan adequate. |
| 2026-04-01 | Pipeline queue over kg_sync_state hack | Proper retry, error tracking, multi-expert routing. |

# Sprint 5 (Phase 9.3) Research: Pipeline Queue Table

## Objective
Replace the `kg_sync_state` key-value table hack with a proper `memory_pipeline_queue` table that supports typed routing, retry logic, and error tracking.

## Current State

### kg_sync_state (the "hack")
```sql
CREATE TABLE IF NOT EXISTS kg_sync_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
INSERT OR IGNORE INTO kg_sync_state VALUES
    ('last_claude_md_sync', '0', 0),
    ('last_memory_md_sync', '0', 0);
```

Used for:
- Tracking last sync time of CLAUDE.md and MEMORY.md files
- NOT actually used as a queue — more of a state tracker

### A-MEM Queue (existing proper queue)
The A-MEM evolution system already has a queue pattern in `db_lifecycle.py`:
- `queue_amem_evolution()` — enqueue a memory for evolution
- `get_pending_amem_queue()` — get pending items (max 7 days old)
- `dequeue_amem()` — mark as processed

This uses `kg_sync_state` table with keys like `amem_pending:{memory_id}`.

### Spec Target (Section 4.1)
```sql
CREATE TABLE memory_pipeline_queue (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    created_at       REAL NOT NULL,
    memory_json      TEXT NOT NULL,
    target_expert    TEXT NOT NULL,
    processed_at     REAL,
    processing_error TEXT,
    retry_count      INTEGER DEFAULT 0
);
CREATE INDEX idx_queue_target ON memory_pipeline_queue(
    target_expert, processed_at);
```

## Design Decisions
1. **Keep kg_sync_state** — it serves a different purpose (tracking sync timestamps). Don't break existing functionality.
2. **Add memory_pipeline_queue** — new table for proper queue semantics.
3. **Migrate A-MEM queue** — move from `amem_pending:` keys in kg_sync_state to proper queue rows with `target_expert = 'amem_evolution'`.
4. **Add worker functions** — enqueue, dequeue, get pending, retry with backoff.
5. **Max retries** — 3 retries with exponential backoff (5s, 25s, 125s).

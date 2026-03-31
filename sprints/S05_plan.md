# Sprint 5 (Phase 9.3) Plan: Pipeline Queue Table

## Scope
Add `memory_pipeline_queue` table per spec. Provide queue CRUD functions. Migrate A-MEM from kg_sync_state hack to proper queue. Keep kg_sync_state for its sync-tracking purpose.

## Files Modified

### 1. `hooks/db.py` — Add DDL + re-exports
- Add `memory_pipeline_queue` CREATE TABLE + index to `_DDL`
- Re-export new functions from `db_lifecycle.py`

### 2. `hooks/db_lifecycle.py` — Queue CRUD functions
- `enqueue_pipeline(session_id, memory_json, target_expert)` → returns queue id
- `get_pending_pipeline(target_expert, limit=10)` → list of unprocessed items
- `complete_pipeline_item(queue_id)` → mark processed_at
- `fail_pipeline_item(queue_id, error_msg)` → increment retry_count, set error
- `get_pipeline_stats()` → counts by target_expert and status
- Update `queue_amem_evolution()` to use pipeline queue instead of kg_sync_state
- Update `get_pending_amem_queue()` to read from pipeline queue
- Update `dequeue_amem()` to use `complete_pipeline_item()`

### 3. `daemon/embedding_daemon.py` — Use new queue in A-MEM processing
- Update `_process_amem_queue()` to use the new pipeline queue functions

## Acceptance Criteria
1. `memory_pipeline_queue` table exists in DDL
2. Enqueue/dequeue/complete/fail functions work
3. A-MEM queue uses pipeline queue (backward compatible)
4. Retry count tracks failures
5. Stats function returns queue health
6. All existing tests pass

## Test Plan (appended to tests/test_phase9.py)
- `TestPipelineQueue` (8 tests): enqueue, get_pending, complete, fail, retry_count, stats, empty queue, target filtering
- `TestAmemQueueMigration` (3 tests): A-MEM queue uses pipeline queue, pending reads from queue, dequeue completes item

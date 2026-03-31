# Sprint 8.3 Research — SessionStart Validity Gates

## Relevant Existing Code
- `hooks/db_memory.py:436-488` — `get_memories_for_session_start()` SQL query filters by `superseded_by IS NULL` and `gc_eligible = 0` but does NOT check `valid_to` or `valid_from`
- `daemon/embedding_daemon.py:440-447` — Search path already checks `valid_to < now` (inconsistency)
- `hooks/session_start.py:85-168` — Main session start flow, calls `db.get_memories_for_session_start()`
- Spec Section 5.1 — SessionStart validity gates: exclude expired `valid_to`, add recent context injection

## Spec Requirements
- Filter loaded standing rules through temporal validity (exclude expired `valid_to`)
- Add semantic recent context injection (top 30 lines, importance >= 5)
- Consistency: session start should apply same validity gates as search

## Patterns to Reuse
- `embedding_daemon.py:440-447` — Same validity gate pattern (valid_to check)
- `db_memory.py` SQL WHERE clause — add `valid_to` and `valid_from` conditions

## Risks
- Adding `valid_from` filter may exclude memories with NULL valid_from (most memories). Must use `(valid_from IS NULL OR valid_from <= ?)`.
- The memories table may not have `valid_to` column in older schemas. Need graceful handling.
- Recent context injection (top 30 lines) needs to not duplicate standing rules already loaded.

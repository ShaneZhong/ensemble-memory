# Sprint 8.3 Plan — SessionStart Validity Gates

## Goal
Add temporal validity filtering to session start memory loading (exclude expired/future memories) and add recent semantic context injection to align with the spec.

## Changes
| File | Change | Est. LOC |
|------|--------|----------|
| hooks/db_memory.py | Add `valid_to`/`valid_from` filters to `get_memories_for_session_start()` SQL, add `get_recent_context()` function | ~40 |
| hooks/session_start.py | Call `get_recent_context()` and append to output | ~15 |
| tests/test_phase8.py | Tests for validity gates and recent context injection | ~80 |

## Design Decisions

### Validity gate SQL
```sql
AND (valid_to IS NULL OR valid_to > ?)
AND (valid_from IS NULL OR valid_from <= ?)
```
This preserves memories with no temporal bounds (vast majority) while filtering truly expired ones.

### Recent context injection
New function `get_recent_context(project, limit=30, min_importance=5)`:
- Returns recent semantic + episodic memories (not corrections/procedural — those are standing rules)
- Ordered by `created_at DESC`
- Excludes memories already returned by `get_memories_for_session_start()`
- Limited to 30 entries to avoid context bloat

### No schema changes
`valid_to` and `valid_from` columns already exist in the memories table (added in Phase 6).

## Acceptance Criteria
- [ ] Expired memories (valid_to < now) excluded from session start
- [ ] Future memories (valid_from > now) excluded from session start
- [ ] Memories with NULL valid_to/valid_from still loaded (majority case)
- [ ] Recent context (semantic+episodic, importance >= 5) injected into session start output
- [ ] No duplicate memories between standing rules and recent context
- [ ] Consistency with search path validity gates

## Test Plan
- Unit: Expired memory excluded from session start
- Unit: Future memory excluded from session start
- Unit: NULL validity bounds still loaded
- Unit: Recent context returns semantic/episodic, not procedural/correction
- Unit: Recent context excludes duplicates from standing rules
- Integration: Full session start with mixed valid/expired memories
- Edge: All memories expired → empty output
- Edge: No recent context available

## Risks
- `valid_to`/`valid_from` column may not exist in test DB. Mitigation: SELECT includes them, test fixtures ensure they exist.
- Adding filters changes existing behavior (currently loads expired memories). This is intentional — spec requires filtering.

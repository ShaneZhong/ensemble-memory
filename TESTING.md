# Ensemble Memory System — Test Results & Known Issues

**Date**: 2026-03-25
**Phase**: Phase 1 Full (SQLite hub + SessionStart loading)
**Test Environment**: Mac Mini M4 16GB, Claude Code v2.1.81, Ollama qwen2.5:3b

---

## Test Results

### Automated Tests (45/45 PASS)

Run: `cd ai_memory/ensemble-memory && python3 tests/test_ensemble_memory.py`

| Component | Tests | Result |
|-----------|-------|--------|
| Triage (regex) | 14 | ALL PASS |
| DB (SQLite hub) | 18 | ALL PASS |
| Write Log (markdown) | 4 | ALL PASS |
| Session Start (loading) | 5 | ALL PASS |
| Integration (end-to-end) | 3 | ALL PASS |

### Live Tests

| Test | Input | Expected | Result | Notes |
|------|-------|----------|--------|-------|
| **Test 1: Correction** | "no, don't use system Python. Always use the .venv Python." | Tier 1 correction captured | **PASS** | importance 7, type correction |
| **Test 2: Decision** | "let's use SQLite for the database instead of PostgreSQL" | Tier 4 decision captured | **PASS** | importance 6, type semantic |
| **Test 3: No false positive** | "can you read the file at ai_memory/progress.md" | No memory entry | **PASS** | Regex correctly found no signals |
| **Test 4: SessionEnd** | N/A | N/A | **SKIPPED** | SessionEnd hook disabled — will become nightly batch job |
| **Test A: SessionStart injection** | Start new session | Memories injected as additionalContext | **PARTIAL** | Hook fires, context injected, but Claude treats it as background info, not directives |
| **Test B: Correction to SQLite** | (pending) | New correction written to both SQLite + markdown | PENDING | |
| **Test C: Supersession live** | (pending) | Old memory marked superseded in SQLite | PENDING | |
| **Test D: Low-signal ignored** | (pending) | No memory entry created | PENDING | |

---

## Known Issues

### HIGH — `additionalContext` is weak

**Problem**: Claude Code's `additionalContext` injection from SessionStart hooks is treated as informational background, not as rules to follow. Even with "MUST FOLLOW" framing, Claude asks "database preference?" instead of automatically choosing SQLite per the standing correction.

**Impact**: Memories are captured but not effectively recalled. The system captures well but retrieval doesn't influence behavior.

**Root cause**: `additionalContext` competes with CLAUDE.md, MEMORY.md, and the user's prompt. It's supplementary, not directive.

**Fix needed**: Query-time retrieval (Phase 2). When user mentions "database", retrieve relevant memories at that moment and inject them into the current context. Session-start injection alone is insufficient.

### MEDIUM — Duplicate memories in markdown

**Problem**: The same memory appears 2-3x in the daily markdown log across different sessions. The Stop hook fires in this session AND sometimes in concurrent sessions, both processing similar turns.

**Impact**: Noisy daily logs, wasted storage.

**Root cause**: The dedup in `write_log.py` checks exact content match, but the LLM sometimes rephrases slightly. Also, the same turn can be processed by multiple concurrent sessions' hooks.

**Fix needed**: Tighter dedup — use content_hash (SHA-256) stored alongside entries, check before writing. Or dedup at the SQLite level (already done) and skip markdown write for known duplicates.

### MEDIUM — Dirty content in SQLite

**Problem**: One memory has `- [importance 8] Actually use SQLite...` as content — the markdown formatting prefix leaked into the extracted content.

**Impact**: Formatting artifacts in stored memories, compounding on each injection cycle.

**Root cause**: The formatted output from `session_start.py` was captured by the Stop hook of a session that logged what it received, feeding the formatted text back through extraction.

**Fix needed**: Strip markdown formatting prefixes from extracted content before storing. Add a content sanitizer in `store_memory.py`.

### LOW — Sessions table empty

**Problem**: `db.record_session()` is never called successfully. The sessions table has no rows despite multiple sessions running.

**Impact**: Cannot track which sessions generated which memories. Minor for now.

**Root cause**: `session_start.py` calls `db.record_session()` but the function signature may have a mismatch, or the call path has an exception being swallowed.

**Fix needed**: Debug the call path, check function signature matches.

### INFO — Extraction latency

**Observation**: Ollama extraction takes 6-13 seconds per turn (average ~9s). One 30-second timeout was hit. Cold model load adds 2-5s on first call per session.

**Impact**: Not user-facing (hook runs async-ish via Stop hook), but affects how quickly memories appear in the log.

**Mitigation**: Already using `format: "json"` constrained generation. Consider keeping model warm with `OLLAMA_KEEP_ALIVE=300s` instead of default 60s.

---

## Extraction Stats Summary

| Metric | Value |
|--------|-------|
| Total extractions attempted | ~15 |
| Success rate | ~90% (13/15) |
| Average latency (success) | ~9.2s |
| Timeouts | 2 (30s limit) |
| JSON validation failures | 0 |
| Retries needed | 0 |

---

## How to Verify

```bash
# Check today's markdown log
cat ~/.ensemble_memory/memory/$(date +%Y-%m-%d).md

# Check SQLite memories
python3 -c "
import sqlite3
conn = sqlite3.connect('\$HOME/.ensemble_memory/memory.db')
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT substr(content,1,60) as content, memory_type, importance, superseded_by FROM memories ORDER BY created_at'):
    print(dict(r))
"

# Check extraction stats
cat ~/.ensemble_memory/logs/extraction_stats.jsonl | tail -10

# Check SessionStart debug log
cat ~/.ensemble_memory/logs/session_start_debug.log

# Run automated tests
cd /Users/shane/Documents/playground/ai_memory/ensemble-memory
python3 tests/test_ensemble_memory.py
```

---

## Architecture (Current State)

```
Claude Code Session
    |
    v
[SessionStart hook] ──> session_start.sh ──> session_start.py
    │                     reads SQLite (importance >= 7)
    │                     outputs {"additionalContext": "..."}
    │                     (injected silently into system prompt)
    |
    v
[Each agent response]
    |
    v
[Stop hook] ──> stop.sh ──> triage.py (< 5ms)
                                |
                          signal? ──no──> done
                                |
                               yes
                                |
                                v
                          extract.py (Ollama qwen2.5:3b, ~9s)
                                |
                                v
                          store_memory.py
                            ├── SQLite insert (db.py)
                            │   ├── content_hash dedup
                            │   ├── temporal metadata
                            │   ├── supersession detection
                            │   └── reinforcement tracking
                            └── Markdown write (write_log.py)
                                ├── daily YYYY-MM-DD.md
                                └── content dedup
```

---

## Next Steps

1. Clean dirty test data from SQLite (`DELETE FROM memories`)
2. Fix content sanitizer to strip markdown formatting
3. Fix sessions table recording
4. Run Tests B, C, D with clean data
5. Phase 2: Query-time retrieval (semantic search via Milvus Lite)

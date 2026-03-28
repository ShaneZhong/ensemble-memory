# Ensemble Memory System — Test Results & Known Issues

**Date**: 2026-03-28 (updated from 2026-03-25)
**Phase**: Phase 3 Complete (Knowledge Graph)
**Test Environment**: Mac Mini M4 16GB, Claude Code v2.1.81, Ollama qwen2.5:3b, sentence-transformers all-MiniLM-L6-v2

---

## Test Results

### Automated Tests (96/96 PASS)

Run: `cd ai_memory/ensemble-memory && python3 tests/test_ensemble_memory.py`

| Component | Tests | Result |
|-----------|-------|--------|
| Triage (regex) | 14 | ALL PASS |
| DB (SQLite hub) | 18 | ALL PASS |
| Write Log (markdown) | 5 | ALL PASS |
| Session Start (loading) | 5 | ALL PASS |
| Integration (end-to-end) | 3 | ALL PASS |
| Embeddings | 9 | ALL PASS |
| Query Retrieval | 5 | ALL PASS |
| Cosine Supersession | 3 | ALL PASS |
| Knowledge Graph (Phase 3) | 26 | ALL PASS |
| Daemon Embed Endpoints (Phase 3) | 8 | ALL PASS |

### Phase 1 Live Tests (2026-03-25)

| Test | Input | Expected | Result | Notes |
|------|-------|----------|--------|-------|
| **Test 1: Correction** | "no, don't use system Python" | Tier 1 captured | **PASS** | importance 7 |
| **Test 2: Decision** | "let's use SQLite for the database" | Tier 4 captured | **PASS** | importance 6 |
| **Test 3: No false positive** | "can you read the file at progress.md" | No memory entry | **PASS** | Regex found no signals |
| **Test A: SessionStart** | Start new session | Memories injected | **PASS** | Context loaded via additionalContext |
| **Test B: Correction to SQLite** | "don't use tabs, use 4 spaces" | Captured to SQLite + markdown | **PASS** | After transcript parser fix |
| **Test C: Supersession** | "use 2 spaces instead of 4" | Memory captured | **PARTIAL** | Captured but Jaccard too strict (0.23 < 0.6) |
| **Test D: No false positive** | "what files are in ai_memory" | No memory stored | **PASS** | After user-only triage fix |

### Phase 2 Live Tests (2026-03-27/28)

| Test | Input | Expected | Result | Notes |
|------|-------|----------|--------|-------|
| **Test 4: Capture + embed** | "don't use print, use logging" | Captured with embedding | **PASS** | After embed-via-daemon fix |
| **Test 5: Retrieve correction** | "how to add debug output" | Logging memory retrieved | **PASS** | Claude responded with `logging` module |
| **Test 6: Unique memory** | "what is the project mascot?" | Retrieves "Koda the blue kangaroo" | **PASS** | Memory only in ensemble-memory, NOT in CLAUDE.md/MEMORY.md. Proves retrieval is 100% from our system |

### Phase 3 Live Tests (2026-03-28)

| Test | Input | Expected | Result | Notes |
|------|-------|----------|--------|-------|
| **Test 7: Entity extraction** | "don't use MySQL, use Redis" | Entities + relationships captured | **PASS** | After connection leak fix |
| **Test 8: Entity dedup** | "use Redis Stack instead of plain Redis" | Redis entity merged, not duplicated | **PASS** | After exact-match-first fix |
| **Test 9: KG-enriched retrieval** | "what cache for graph data?" | Claude cited Zephyr + Moonbeam Protocol (fictional, KG-only) | **PASS** | After keyword stemming fix |
| **Test 10: 2-hop traversal** | "use Python for cache wrapper" | Python → Zephyr → Moonbeam chain traversed | **PASS** | |
| **Test 11: Cold-start bootstrap** | `bootstrap_from_files(CLAUDE.md, MEMORY.md)` | 28 entities, 27 relationships extracted | **PASS** | After chunking fix |

---

## Known Issues

### HIGH — `additionalContext` is weak

**Problem**: Claude Code's `additionalContext` injection from SessionStart hooks is treated as informational background, not as rules to follow. Even with "MUST FOLLOW" framing, Claude asks "database preference?" instead of automatically choosing SQLite per the standing correction.

**Impact**: Memories are captured but not effectively recalled. The system captures well but retrieval doesn't influence behavior.

**Root cause**: `additionalContext` competes with CLAUDE.md, MEMORY.md, and the user's prompt. It's supplementary, not directive.

**Fix (Phase 2)**: Query-time retrieval via the embedding daemon. When user mentions "database", relevant memories are retrieved at that moment and injected into the current context. Phase 3 further enriches this with KG neighborhood context.

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
[User types a prompt]
    |
    v
[UserPromptSubmit hook] ──> user_prompt_submit.sh ──> HTTP POST /search (daemon)
    │                          extracts keywords from prompt
    │                          ├── cosine similarity search (embeddings)
    │                          └── KG neighborhood lookup
    │                               (keyword → FTS5 → BFS 2-hop)
    │                          injects memories + KG context via hookSpecificOutput
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
                          (returns memories + entities + relationships)
                                |
                                v
                          store_memory.py
                            ├── SQLite insert (db.py)
                            │   ├── content_hash dedup
                            │   ├── temporal metadata
                            │   ├── supersession detection
                            │   └── reinforcement tracking
                            ├── KG upsert (kg.py)
                            │   ├── entity resolution + dedup
                            │   └── relationship edge insert
                            └── Markdown write (write_log.py)
                                ├── daily YYYY-MM-DD.md
                                └── content dedup
```

---

## Next Steps

1. Phase 4: Nightly batch processor (async transcript scan for missed turns)
2. Phase 5: Public skill installer (`/plugin install ensemble-memory`)
3. Known issues tracked below still apply to Phase 3

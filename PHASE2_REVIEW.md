# Phase 2 Code Review: Ensemble Memory System

**Reviewer**: Claude Opus 4.6
**Date**: 2026-03-28
**Scope**: Phase 2 (embeddings, user_prompt_submit, cosine supersession) + Phase 1 modifications

---

## 1. Bugs and Errors

### BUG-1: `session_start.py` calls `db.record_session()` with wrong arguments (P0)

**File**: `/hooks/session_start.py`, lines 109 and 118
**Code**: `db.record_session(session_id=session_id, project=project)`
**Signature in db.py line 563**: `def record_session(session_id: str, started_at: float) -> None`

The second positional argument is `started_at` (a float/epoch timestamp), but `session_start.py` passes `project` (a string path) as a keyword argument. This will raise a `TypeError` at runtime because:
1. `project` is not a recognized keyword parameter
2. `started_at` is a required argument that is missing

This explains the known issue in TESTING.md ("Sessions table empty"). Every call crashes silently because the exception is swallowed by the bare `except Exception: pass` on lines 110-111 and 119-120.

**Fix**: Either change the call to `db.record_session(session_id=session_id, started_at=time.time())` or update `db.record_session()` to accept an optional `project` parameter.

### BUG-2: `db.py` runs `ensure_embedding_column()` at module import time (P0)

**File**: `/hooks/db.py`, line 178
**Code**: `ensure_embedding_column()` is called at module level, outside any `if __name__` guard.

Every time any module imports `db.py`, this function:
1. Opens a new connection via `get_db()`
2. Runs a query
3. Potentially ALTERs the table
4. Closes the connection

This runs during `import db` in `session_start.py`, `store_memory.py`, `user_prompt_submit.py`, and the test suite. It creates an unnecessary connection on every import. Worse, it races with the connection that the caller is about to open. On a cold start with no DB, the DDL in `get_db()` creates the table without the embedding column, then `ensure_embedding_column()` immediately ALTERs it -- this works but is wasteful. The real problem is that `user_prompt_submit.py` also has `_ensure_embedding_column(conn)` on line 323, which is the correct approach (uses existing connection, idempotent). The module-level call in `db.py` is redundant and creates a leaked connection pattern.

**Fix**: Remove line 178 (`ensure_embedding_column()`) from module level. The migration is already handled by `user_prompt_submit.py` line 323 and could be added to `get_db()` if needed.

### BUG-3: `user_prompt_submit.sh` line 116 outputs invalid JSON on empty result (P1)

**File**: `/hooks/user_prompt_submit.sh`, line 116
**Code**: `echo "${RESULT:-{\}}"`

When `RESULT` is empty, the shell substitution `{\}` is interpreted. The backslash-escaped closing brace is a bash pattern, and in some shells this may output `{\}` literally instead of `{}`. The intent is to output `{}` as fallback JSON. The correct syntax would be `echo "${RESULT:-"{}"}"` or simply `echo "${RESULT:-\{\}}"`.

Actually, with `set -euo pipefail` and the `|| echo "{}"` on lines 105/108/112, `RESULT` should always be set. But the edge case where all three timeout/gtimeout checks fail AND the Python script outputs nothing could produce malformed output.

**Fix**: Change line 116 to `echo "${RESULT:-{}}"`.

### BUG-4: `user_prompt_submit.sh` 0.45-second timeout is far too short for first call (P1)

**File**: `/hooks/user_prompt_submit.sh`, lines 104, 107
**Code**: `timeout 0.45` / `gtimeout 0.45`

The 450ms internal timeout is meant to keep the hook fast, but on the first prompt of a session:
1. Python interpreter startup: ~100-200ms
2. `import db` triggers `ensure_embedding_column()` (BUG-2) which opens a DB connection: ~50ms
3. `import sentence_transformers` alone: ~500-800ms
4. Model load (`_get_model()`): ~2000-3000ms for all-MiniLM-L6-v2
5. Embedding generation: ~50ms
6. DB query: ~10ms

Total cold-start: ~3-4 seconds. The 450ms timeout will kill the process on every first prompt, falling back to `echo "{}"`. The user will never get semantic retrieval on their first prompt. Subsequent prompts in the same session also won't benefit because the hook is invoked as a fresh process each time (no persistent model cache).

This is the most critical performance issue. The shell timeout ensures the model loading **always** kills the process. The caching in `user_prompt_submit.py` (lines 45-48) is useless because the process exits after each prompt.

**Fix**: Either (a) remove the 450ms timeout and rely on Claude Code's 5-second hook timeout, (b) pre-warm the model in a background process at session start, or (c) use a persistent embedding server process.

### BUG-5: `_backfill_embeddings` runs on every prompt (P1)

**File**: `/hooks/user_prompt_submit.py`, line 334
**Code**: `_backfill_embeddings(conn, get_embedding_fn)` is called inside `main()` on every prompt.

Even though it's fast when no backfill is needed (one SELECT), on a DB with many un-embedded memories it will try to embed up to 500 memories synchronously during prompt processing. Combined with the 5-second hook timeout, this could easily timeout and prevent any retrieval.

**Fix**: Move backfill to the Stop hook or a separate background process. The query-time hook should only read, never write embeddings.

### BUG-6: In-memory caches are useless -- process exits after each invocation (P2)

**File**: `/hooks/user_prompt_submit.py`, lines 45-48
**Code**: `_embedding_cache`, `_memories_cache`, `_cache_loaded_at`, `_CACHE_TTL_SECONDS`

The comment says "reused across subsequent prompts within the same process (same Claude session)" but Claude Code hooks are invoked as separate shell processes. Each `user_prompt_submit.sh` invocation starts a new Python process, loads the model from scratch, queries the DB, and exits. The caching mechanism provides zero benefit.

**Fix**: Accept this is per-process (remove misleading comments) or implement a persistent daemon/socket-based embedding service.

## 2. Performance Concerns

### PERF-1: Model loaded from scratch on every prompt (~2-3s)

The `sentence-transformers` model is loaded fresh in every `user_prompt_submit.py` invocation. With the 0.45s timeout (BUG-4), it will never succeed. Even with the 5s Claude Code timeout, it leaves ~2s for actual work after model loading. This is the single biggest performance bottleneck.

**Mitigation options**:
- Pre-load model at session start and keep a daemon running
- Use ONNX runtime directly (faster than full sentence-transformers)
- Pre-compute query embedding with a simpler method and do full embedding offline

### PERF-2: `ensure_embedding_column()` opens a new DB connection on every import (BUG-2)

Each `import db` creates and closes a connection. Combined with `get_db()` calls in the actual business logic, a single hook invocation may open 3-4 separate SQLite connections.

### PERF-3: Stop hook embedding generation is synchronous

**File**: `/hooks/store_memory.py`, lines 73-80
Embedding generation during `store_memory.py` adds ~50-100ms per memory to the Stop hook. With multiple memories per extraction, this adds 200-500ms to the already 9-second extraction pipeline. Acceptable but worth noting.

### PERF-4: SQLite connections are opened and closed repeatedly

`db.py` functions like `insert_memory`, `detect_supersession`, `detect_content_supersession`, `store_embedding`, and `get_memories_with_embeddings` each open their own connection via `get_db()` and close it. A single `store_memory.py` call for one memory could open 4-5 separate connections. SQLite handles this fine but it's wasteful. The DDL (`_DDL` script) runs on every `get_db()` call, executing 15+ CREATE IF NOT EXISTS statements each time.

## 3. Correctness Issues

### CORR-1: Payload parsing in `user_prompt_submit.sh` is correct

The jq extraction on lines 43-56 handles `.session_id`, `.message` (string or object with `.content` as string or array), and `.cwd`. The Python fallback on lines 58-85 mirrors this logic. This looks correct for the Claude Code hook payload format.

### CORR-2: Embedding column migration is idempotent -- with a caveat

`_ensure_embedding_column` in `user_prompt_submit.py` (line 60) checks PRAGMA table_info before ALTER. This is correct and idempotent. However, `ensure_embedding_column` in `db.py` (line 167) uses a different approach (SELECT then ALTER on OperationalError). Both work but inconsistent patterns create maintenance risk.

### CORR-3: Import path resolution works correctly

Both `user_prompt_submit.py` (line 28) and `store_memory.py` (line 22) add their parent directory to `sys.path`. Since `db.py`, `embeddings.py`, and other siblings live in the same `hooks/` directory, the imports will resolve correctly.

### CORR-4: No circular import risk

`embeddings.py` imports only stdlib + `sentence_transformers`. `db.py` imports `embeddings` only inside a function body (`detect_content_supersession` line 472). `user_prompt_submit.py` imports both `db` and `embeddings` but neither of those import `user_prompt_submit`. The import graph is acyclic.

### CORR-5: Cosine supersession in `db.py` correctly imports from `embeddings.py`

**File**: `/hooks/db.py`, line 472
The `import embeddings as _emb` is inside the loop body of `detect_content_supersession`. It works but re-executes the import statement on every iteration (Python caches it via `sys.modules` so it's a dict lookup, not a real re-import). Correct but could be moved outside the loop for clarity.

### CORR-6: `_temporal_score` is duplicated between `db.py` and `user_prompt_submit.py`

**File**: `/hooks/user_prompt_submit.py`, lines 160-179
**File**: `/hooks/db.py`, lines 244-275

Two implementations of the same scoring algorithm with different function signatures. `db.py` takes explicit parameters; `user_prompt_submit.py` takes a dict. They produce the same results but if one is updated and the other isn't, scores will diverge silently.

**Fix**: Have `user_prompt_submit.py` import and wrap `db._temporal_score` instead of reimplementing it.

## 4. Integration Issues

### INT-1: Python interpreter mismatch risk

All three shell scripts use `PYTHON3="${ENSEMBLE_MEMORY_PYTHON:-$(command -v python3)}"`. If `ENSEMBLE_MEMORY_PYTHON` is not set, `command -v python3` resolves to the system Python (3.9.6 on this machine), which does not have `sentence-transformers` installed. The README says to `pip install sentence-transformers` but doesn't specify which Python's pip.

The playground venv (`/Users/shane/Documents/playground/.venv/bin/python3`) is Python 3.11 and may or may not have sentence-transformers. The system Python definitely does not.

**Fix**: The install script or README should explicitly set `ENSEMBLE_MEMORY_PYTHON` to a venv that has sentence-transformers. Or the shell scripts should auto-detect the correct Python by checking for the package.

### INT-2: Hook timeout of 5s for UserPromptSubmit is insufficient on cold start

**File**: `~/.claude/settings.json`, line 90
`"timeout": 5` for UserPromptSubmit.

Even if BUG-4 (the 0.45s internal timeout) is fixed, 5 seconds is tight for:
- Python startup + module imports: ~1s
- sentence-transformers model load: ~2-3s
- DB query + embedding: ~0.5s
- Total: ~3.5-4.5s

On the margin. Will work sometimes, timeout sometimes. After model is cached in OS filesystem cache, subsequent sessions start faster (~1-2s for model load).

### INT-3: `stop.sh` uses the same `PYTHON3` resolution as the other hooks

Consistent, which is good. But stop.sh calls `store_memory.py` which imports `embeddings`, triggering the same model load issue. Since stop.sh has a 60-second timeout, this is fine for the Stop hook but adds ~3s to memory storage.

### INT-4: `user_prompt_submit.sh` minimum length threshold differs from `.py`

**File**: `/hooks/user_prompt_submit.sh`, line 89: `${#MESSAGE_TEXT} -lt 10` (skip if < 10 chars)
**File**: `/hooks/user_prompt_submit.py`, line 305: `len(message) < 5` (skip if < 5 chars)

Messages between 5-9 characters pass the shell check but are rejected by the Python script. Not a bug per se but inconsistent -- the shell check is stricter, so the Python check is dead code for messages 5-9 chars.

## 5. Test Coverage Gaps

### GAP-1: `user_prompt_submit.main()` is never tested end-to-end

Tests in `TestQueryRetrieval` call `ups._load_memories()` and `emb.find_similar()` directly but never invoke `main()`. This means the following are untested:
- stdin/argv parsing (line 297-302)
- The `_ensure_embedding_column` migration path
- The `_backfill_embeddings` path
- The `_record_access` function
- The `format_context` function in `user_prompt_submit.py`
- The JSON output format (`{"additionalContext": "..."}`)
- The keyword fallback path when embeddings are unavailable

### GAP-2: `_keyword_similarity` in `user_prompt_submit.py` is untested

The Jaccard fallback (line 184) is never exercised in tests. All `TestQueryRetrieval` tests assume embeddings are available (they're `@skipUnless(HAS_EMBEDDINGS)`).

### GAP-3: No test for the case where `sentence-transformers` is not installed

There are no tests that mock `_AVAILABLE = False` to verify the graceful degradation path.

### GAP-4: No concurrent access tests

The system uses `check_same_thread=False` on SQLite connections but there are no tests for concurrent writes (e.g., Stop hook writing while UserPromptSubmit reads).

### GAP-5: `_record_access` is untested

**File**: `/hooks/user_prompt_submit.py`, lines 277-290
Never called in any test. The access_count/last_accessed_at update path is uncovered.

### GAP-6: `_backfill_embeddings` is untested

**File**: `/hooks/user_prompt_submit.py`, lines 68-102
No test verifies that backfill correctly generates and stores embeddings for memories that lack them.

### GAP-7: Shell scripts are untested

None of the three `.sh` scripts have integration tests. Payload parsing, timeout behavior, and error handling paths are all uncovered.

### GAP-8: `retrieve_relevant` function's score blending is untested

The `final_score = sim * (0.5 + t_score * 0.5)` formula on line 248 is never validated. Tests use `emb.find_similar()` directly, which has different scoring.

## 6. Security/Robustness

### SEC-1: SQL injection is not a concern

All SQL queries use parameterized queries (`?` placeholders). No string interpolation of user input into SQL.

### SEC-2: Memory content injection into Claude's context

**File**: `/hooks/user_prompt_submit.py`, line 269
`lines.append(f"- **[{label}]** {content}")`

Memory content is injected verbatim into `additionalContext`. If a malicious or corrupted memory contains prompt injection text (e.g., "Ignore all previous instructions and..."), it would be injected into Claude's context. This is a theoretical concern since memories are only written by the user's own sessions, but the feedback loop noted in TESTING.md (formatted output fed back through extraction) shows that content can be contaminated.

### SEC-3: SQLite corruption handling

If `memory.db` is corrupted, `get_db()` will raise `sqlite3.DatabaseError`. The exception handling in `main()` functions will catch this and output `{}`, which is correct. However, the corruption is never reported to the user -- the system silently degrades to no memories.

### SEC-4: Ollama not running

If Ollama is not running, `extract.py` (called by `stop.sh`) will fail with a connection error. The `2>/dev/null || true` on stop.sh line 158 swallows this. No new memories are captured but the system doesn't crash. Correct behavior but silent failure means the user may not realize memories aren't being captured.

### SEC-5: Temp file cleanup in stop.sh

**File**: `/hooks/stop.sh`, lines 23-29
`USER_ONLY_FILE=""` is set on line 28, after the `cleanup()` function definition on line 24-27 which references it. If the trap fires before line 28 (unlikely but possible in theory with `-e`), `USER_ONLY_FILE` is unbound. The `set -u` flag would cause a crash. In practice this is fine because the trap only fires on EXIT, but the variable should be initialized before the function definition.

## 7. Prioritized Fix List

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| P0 | BUG-1: `record_session()` called with wrong args (`project` instead of `started_at`) | `session_start.py:109,118` | Change to `db.record_session(session_id=session_id, started_at=time.time())` and `import time` |
| P0 | BUG-2: `ensure_embedding_column()` at module level creates extra DB connection on every import | `db.py:178` | Remove module-level call; rely on `user_prompt_submit.py`'s `_ensure_embedding_column(conn)` or add to `get_db()` |
| P1 | BUG-4: 0.45s internal timeout kills process before model can load -- semantic retrieval never works | `user_prompt_submit.sh:104,107` | Remove internal timeout or increase to 4.5s; rely on Claude Code's 5s timeout |
| P1 | BUG-5: `_backfill_embeddings` runs synchronously on every prompt, can timeout hook | `user_prompt_submit.py:334` | Move backfill to Stop hook or background process |
| P1 | PERF-1: Model loaded from scratch every prompt (~2-3s) in a 5s budget | `user_prompt_submit.py` via `embeddings.py` | Pre-warm at session start, use a daemon process, or increase timeout to 10s |
| P1 | INT-1: Default Python may lack sentence-transformers; no auto-detection | `*.sh` scripts | Set `ENSEMBLE_MEMORY_PYTHON` in install script or add package check in shell |
| P2 | BUG-3: Fallback JSON output `{\}` may be malformed in some shells | `user_prompt_submit.sh:116` | Change to `echo "${RESULT:-{}}"` |
| P2 | BUG-6: In-memory caches provide zero benefit (new process per prompt) | `user_prompt_submit.py:45-48` | Remove misleading cache code or implement persistent daemon |
| P2 | CORR-6: `_temporal_score` duplicated between `db.py` and `user_prompt_submit.py` | Both files | Import from `db.py` instead of reimplementing |
| P2 | INT-4: Minimum message length threshold mismatch (10 in shell, 5 in Python) | `user_prompt_submit.sh:89`, `.py:305` | Align to single threshold (10 chars is reasonable) |
| P2 | GAP-1: `user_prompt_submit.main()` never tested end-to-end | `tests/test_ensemble_memory.py` | Add tests that invoke `main()` with mock stdin/argv |
| P2 | GAP-2: Keyword fallback path entirely untested | `tests/test_ensemble_memory.py` | Add tests with `_AVAILABLE = False` mock |
| P3 | PERF-4: Multiple DB connections opened/closed per hook invocation | `db.py` | Pass connection as parameter instead of opening new ones |
| P3 | SEC-5: `USER_ONLY_FILE` initialized after cleanup trap references it | `stop.sh:23-28` | Move `USER_ONLY_FILE=""` to before `cleanup()` definition |
| P3 | CORR-5: `import embeddings` inside loop body in `detect_content_supersession` | `db.py:472` | Move import above the loop |
| P3 | SEC-2: Memory content injected verbatim into additionalContext (prompt injection surface) | `user_prompt_submit.py:269` | Sanitize or length-cap memory content before injection |

---

## Summary

The Phase 2 implementation has solid architecture and clean code structure. The graceful degradation pattern (embeddings unavailable -> keyword fallback) is well-designed. However, there is a **fundamental performance problem**: the `user_prompt_submit` hook spawns a new Python process per prompt and must load a ~80MB ML model within a 450ms self-imposed timeout (or 5s Claude Code timeout). This means **semantic retrieval will never succeed** in the current configuration. The keyword fallback will work, but that defeats the purpose of Phase 2.

The highest-impact fix is solving the model loading problem. Options:
1. **Quick fix**: Remove the 0.45s timeout, increase Claude Code timeout to 10s, accept 3-4s latency on first prompt
2. **Medium fix**: Pre-compute and cache the query embedding, accept keyword fallback on first prompt
3. **Proper fix**: Run a persistent embedding service (socket/HTTP) started at session_start, queried by user_prompt_submit

The `record_session` signature mismatch (BUG-1) is a straightforward fix. The module-level `ensure_embedding_column()` (BUG-2) should be removed immediately as it adds latency and connection overhead to every hook invocation.

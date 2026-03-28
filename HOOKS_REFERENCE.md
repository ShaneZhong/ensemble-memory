# Claude Code Hooks Reference

**Critical**: Always refer to this document when modifying hooks. Do NOT guess the payload format — it has caused multiple production bugs.

Official docs: https://code.claude.com/docs/en/hooks.md

---

## Hook Payload Formats (Input via stdin)

### SessionStart

```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/session.jsonl",
  "cwd": "/Users/shane/Documents/playground",
  "hook_event_name": "SessionStart",
  "source": "startup | resume | clear | compact",
  "model": "claude-opus-4-6"
}
```

### UserPromptSubmit

```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/session.jsonl",
  "cwd": "/Users/shane/Documents/playground",
  "permission_mode": "default",
  "hook_event_name": "UserPromptSubmit",
  "prompt": "the user's actual message text"
}
```

**WARNING**: The user's text is in `"prompt"`, NOT `"message"`. This caused a bug where query-time retrieval silently returned nothing for every prompt.

### Stop

```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/session.jsonl",
  "cwd": "/Users/shane/Documents/playground",
  "permission_mode": "default",
  "hook_event_name": "Stop",
  "stop_hook_active": false,
  "last_assistant_message": "Claude's last response text"
}
```

**WARNING**: Always check `stop_hook_active`. If `true`, Claude is already continuing from a previous Stop hook — skip processing to prevent infinite loops.

---

## Hook Response Formats (Output via stdout)

### SessionStart

```json
{"additionalContext": "Context text injected into Claude's prompt"}
```

Top-level `additionalContext`. Returns `{}` if no context to inject.

### UserPromptSubmit

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "Context text injected before Claude responds"
  }
}
```

**WARNING**: `additionalContext` MUST be inside `hookSpecificOutput`, NOT at the top level. Top-level `additionalContext` is silently ignored for UserPromptSubmit. This caused a bug where retrieved memories were never injected.

### Stop

No stdout output needed for capture-only hooks. Just `exit 0`.

To block Claude from stopping: `{"decision": "block", "reason": "explanation"}` (requires a reason).

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — stdout JSON is parsed |
| 2 | Blocking error — stderr fed back to Claude |
| Other | Non-blocking error — stderr shown in verbose mode, execution continues |

---

## Common Fields (All Hooks)

Every hook payload includes:
- `session_id` — unique session identifier
- `cwd` — current working directory
- `hook_event_name` — the event that triggered the hook
- `transcript_path` — path to the `.jsonl` conversation log

---

## Lessons Learned

1. **Always inspect raw payloads** when a hook isn't working. Add a debug hook that logs `cat` stdin to a file.
2. **UserPromptSubmit uses `prompt`** not `message` — this is different from what you might expect.
3. **UserPromptSubmit response wraps in `hookSpecificOutput`** — SessionStart does NOT.
4. **Stop hook has `last_assistant_message`** — you can use this instead of parsing the transcript for the last response.
5. **The transcript JSONL `message` field can be a string, dict, or list** — always handle all three formats.
6. **Tool results appear as user-type messages** with `{"type": "tool_result"}` content blocks — filter these out when looking for actual user text.

---

## Embedding Daemon Endpoints (port 9876)

The daemon is called by `user_prompt_submit.py` and `kg.py`. All endpoints accept/return JSON.

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/search` | POST | `{"query": "...", "project": "...", "top_k": 5}` | `{"results": [...], "kg_context": "..."}` |
| `/embed` | POST | `{"text": "..."}` | `{"embedding": [0.1, ...]}` (384-dim) |
| `/embed_batch` | POST | `{"texts": ["...", "..."]}` | `{"embeddings": [[...], [...]]}` |
| `/health` | GET | — | `{"status": "ok", "model": "..."}` |
| `/invalidate_cache` | POST | `{}` | `{"ok": true}` |

**Error codes**: 400 (missing fields), 503 (model not loaded), 200 (success).

---

## settings.json Hook Registration

```json
{
  "hooks": {
    "Stop": [{"hooks": [{"type": "command", "command": "/path/to/stop.sh", "timeout": 60}]}],
    "SessionStart": [{"hooks": [{"type": "command", "command": "/path/to/session_start.sh", "timeout": 10}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "/path/to/user_prompt_submit.sh", "timeout": 5}]}]
  }
}
```

Note: timeout is in **seconds** (not milliseconds).

# Ensemble Memory System

A 100% free, local memory system for AI coding agents. Captures corrections, decisions, and insights from your sessions automatically — then retrieves them semantically when relevant. Zero API cost, runs entirely on your machine.

## How It Works

```
Claude Code session
    |
    v
[SessionStart hook] ──> Load standing rules (importance >= 7) ──> Inject into context
    |
    v
[User types a prompt]
    |
    v
[UserPromptSubmit hook] ──> Query daemon ──> Find similar memories ──> Inject relevant ones
    |                        (~50ms)          (cosine similarity)       via hookSpecificOutput
    v
[Claude responds]
    |
    v
[Stop hook] ──> [Regex triage < 5ms] ──> signal? ──no──> done
                                              |
                                             yes
                                              |
                                              v
                                      [Ollama qwen2.5:3b ~9s]
                                              |
                                              v
                                   [SQLite + Markdown write]
                                              |
                                              v
                                   [Daemon /embed + /invalidate_cache]
```

**Three hooks + one daemon:**
1. **SessionStart** — loads standing rules as baseline context
2. **UserPromptSubmit** — queries the embedding daemon for semantically relevant memories
3. **Stop** — captures new corrections and decisions after each response
4. **Embedding daemon** — persistent background process (port 9876) that loads the ML model once (~200MB), serves embedding + search requests in ~50ms

## What Gets Captured

- Corrections: "no, don't use MySQL, use PostgreSQL"
- Decisions: "let's use SQLite for the database"
- Procedural rules: derived patterns from repeated corrections

## What Doesn't Get Captured

- Ordinary requests: "read this file", "run the tests"
- Code output, tool results, assistant responses

## Requirements

- **Ollama** with `qwen2.5:3b` model (~1.9GB)
- **Python 3.10+**
- **sentence-transformers** (`pip install sentence-transformers`) — for semantic search
- **Claude Code** with hooks support

## Install

```bash
git clone https://github.com/ShaneZhong/ensemble-memory.git
cd ensemble-memory
pip install sentence-transformers
./install.sh

# Start the embedding daemon
bash daemon/daemon_ctl.sh start
```

`install.sh` will:
1. Create `~/.ensemble_memory/` directories
2. Check Ollama is installed and pull `qwen2.5:3b` if needed
3. Copy default config to `~/.ensemble_memory/config.toml`
4. Register hooks in Claude Code's `~/.claude/hooks.json`

The embedding daemon auto-starts when hooks detect it's down, but you can manage it manually:
```bash
bash daemon/daemon_ctl.sh start    # Start daemon
bash daemon/daemon_ctl.sh stop     # Stop daemon
bash daemon/daemon_ctl.sh status   # Check status
bash daemon/daemon_ctl.sh restart  # Restart
```

Then add the hooks to your `~/.claude/settings.json`:
```json
{
  "hooks": {
    "Stop": [{"hooks": [{"type": "command", "command": "<path>/hooks/stop.sh", "timeout": 60}]}],
    "SessionStart": [{"hooks": [{"type": "command", "command": "<path>/hooks/session_start.sh", "timeout": 10}]}],
    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "<path>/hooks/user_prompt_submit.sh", "timeout": 5}]}]
  }
}
```

## What Gets Created

```
~/.ensemble_memory/
├── config.toml                 # Configuration (model, timeouts, etc.)
├── memory.db                   # SQLite database (structured memories + embeddings)
├── memory/
│   └── YYYY-MM-DD.md           # Daily logs (human-readable, source of truth)
└── logs/
    ├── extractions/            # Raw extraction JSON (debug/reprocess)
    ├── extraction_stats.jsonl  # Success rate tracking
    └── missed_turns.jsonl      # Turns that timed out
```

## Configuration

All settings via environment variables or `~/.ensemble_memory/config.toml`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENSEMBLE_MEMORY_DIR` | `~/.ensemble_memory/` | Data directory |
| `ENSEMBLE_MEMORY_LOGS` | `~/.ensemble_memory/memory/` | Daily log output |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `ENSEMBLE_MEMORY_MODEL` | `qwen2.5:3b` | Ollama model for extraction |
| `ENSEMBLE_MEMORY_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `ENSEMBLE_MEMORY_TIMEOUT` | `30` | Extraction timeout (seconds) |
| `ENSEMBLE_MEMORY_PYTHON` | `python3` | Python interpreter path |

## Features

### Capture
- Regex triage gate (<5ms) — only fires LLM on memory-worthy turns
- Ollama `format: "json"` constrained generation for reliable output
- SQLite hub with full temporal metadata (decay rates, stability, supersession chains)
- Content-hash dedup at SQLite level (no duplicate memories)
- Embedding generated on insert (all-MiniLM-L6-v2, 384-dim)
- Reinforcement tracking (repeated patterns flag promotion candidates for CLAUDE.md)
- Human-readable markdown daily logs as source of truth

### Recall
- **SessionStart**: loads standing corrections and rules (importance >= 7) as baseline context
- **UserPromptSubmit**: semantic query-time retrieval — embeds the user's prompt, finds similar memories via cosine similarity, injects only relevant ones as context BEFORE Claude responds
- Temporal decay scoring (ACT-R Petrov + Ebbinghaus forgetting curve) weights fresher memories higher

### Supersession
- **Structured**: same subject + predicate → old fact automatically superseded
- **Cosine similarity** (>= 0.85): semantically similar corrections auto-supersede (Phase 2)
- **Jaccard fallback** (>= 0.6): word-overlap supersession when no embeddings available
- Superseded memories are retained for history but excluded from retrieval

### Temporal Model
- **Ebbinghaus decay**: `strength = exp(-lambda_eff * t_days)` where `lambda_eff = decay_rate * (1 - stability * 0.8)`
- **ACT-R Petrov**: `B_i = ln(n/(1-d)) - d*ln(t)` with d=0.5
- **Stability**: `(importance - 1) / 9.0` — high-importance memories decay slower
- **Per-type decay rates**: procedural=0.01, correction=0.05, semantic=0.10, episodic=0.16
- Memories accessed more often decay slower (access_count boosts ACT-R activation)

## Architecture

Built from a 6-round Delphi process with 6 expert AI agents, each specializing in a memory dimension:

1. **Semantic** — Embedding + BM25 hybrid search (Memsearch-inspired)
2. **Hybrid RAG** — Decision extraction + cross-encoder reranking (ClawMem-inspired)
3. **Contextual** — Chunk enrichment before embedding (Anthropic Contextual Retrieval)
4. **Temporal** — Recency decay, Ebbinghaus forgetting curves, ACT-R activation
5. **Knowledge Graph** — Entity-relationship graphs, A-MEM Zettelkasten
6. **Cognitive** — Memory triage, classification, forgetting policies

Full design: [final_design.md](../synthesis/final_design.md) (1,931 lines)

### File Layout

```
ensemble-memory/
├── daemon/
│   ├── embedding_daemon.py    # Persistent HTTP server (port 9876, ~200MB RAM)
│   └── daemon_ctl.sh          # start/stop/restart/status management
├── hooks/
│   ├── db.py                  # SQLite hub (schema, temporal scoring, supersession)
│   ├── embeddings.py          # Sentence-transformers wrapper (all-MiniLM-L6-v2, 384-dim)
│   ├── triage.py              # Regex signal detection (< 5ms)
│   ├── extract.py             # Ollama qwen2.5:3b caller with JSON validation + retry
│   ├── write_log.py           # Markdown daily log writer with dedup
│   ├── store_memory.py        # SQLite + embedding + markdown orchestrator
│   ├── session_start.sh/py    # SessionStart hook — load standing rules
│   ├── stop.sh                # Stop hook — capture corrections + decisions
│   ├── user_prompt_submit.sh/py  # UserPromptSubmit hook — thin HTTP client to daemon
│   └── prompts/extraction.txt # LLM prompt template
├── tests/
│   └── test_ensemble_memory.py  # 62 tests (Phase 1 + Phase 2)
├── config/default_config.toml
├── install.sh
├── HOOKS_REFERENCE.md         # Critical: hook payload/response format docs
├── PHASE2_REVIEW.md           # Code review findings
└── TESTING.md
```

## Tests

```bash
python3 tests/test_ensemble_memory.py
```

62 tests covering:
- Triage: 14 tests (all regex patterns, false positive rejection, case sensitivity, user-only scanning)
- DB: 18 tests (CRUD, temporal scoring, supersession, dedup, reinforcement, confidence fields)
- Write Log: 4 tests (file creation, format, dedup)
- Session Start: 5 tests (loading, filtering, grouping, format)
- Integration: 3 tests (store + load roundtrip, supersession, full pipeline)
- Embeddings: 9 tests (generation, dimension, similarity, batch, find_similar)
- Query Retrieval: 5 tests (relevant/irrelevant queries, multiple results, embedding storage)
- Cosine Supersession: 3 tests (similar supersedes, unrelated doesn't, Jaccard fallback)

## Roadmap

- [x] Phase 1: Capture pipeline (regex triage → Ollama extraction → SQLite + markdown)
- [x] Phase 1: SessionStart loading (importance >= 7 standing rules)
- [x] Phase 2: Semantic search (sentence-transformers embeddings, cosine similarity)
- [x] Phase 2: Query-time retrieval (UserPromptSubmit hook via embedding daemon)
- [x] Phase 2: Cosine supersession (replaces Jaccard for embedded memories)
- [x] Phase 2: Persistent embedding daemon (port 9876, auto-start, 30min idle shutdown)
- [ ] Phase 3: Knowledge graph (SQLite adjacency tables, entity extraction)
- [ ] Phase 4: Nightly batch processor (async transcript scan)
- [ ] Phase 5: Public skill installer (`/plugin install ensemble-memory`)

## Uninstall

```bash
# Remove hooks from Claude Code settings
# (edit ~/.claude/settings.json and remove Stop/SessionStart/UserPromptSubmit entries)

# Remove data
rm -rf ~/.ensemble_memory/
```

## License

MIT

# Ensemble Memory System

A 100% free, local memory system for AI coding agents. Captures corrections, decisions, and insights from your sessions automatically — zero API cost, runs entirely on your machine.

## How It Works

```
Claude Code session
    |
    v
[Stop hook] ──> [Regex triage < 5ms] ──> signal? ──no──> done
                                              |
                                             yes
                                              |
                                              v
                                      [Ollama qwen2.5:3b]
                                              |
                                              v
                                   [SQLite + Markdown write]

[SessionStart hook] ──> Load standing rules & corrections ──> Inject into context
```

**What gets captured:**
- Corrections: "no, don't use MySQL, use PostgreSQL"
- Decisions: "let's use SQLite for the database"
- Procedural rules: derived patterns from repeated corrections

**What doesn't get captured:**
- Ordinary requests: "read this file", "run the tests"
- Code output, tool results, assistant responses

## Requirements

- **Ollama** with `qwen2.5:3b` model (~1.9GB)
- **Python 3.10+** (stdlib only, no pip dependencies)
- **Claude Code** with hooks support

## Install

```bash
git clone https://github.com/shaneholloman/ensemble-memory.git
cd ensemble-memory
./install.sh
```

`install.sh` will:
1. Create `~/.ensemble_memory/` directories
2. Check Ollama is installed and pull `qwen2.5:3b` if needed
3. Copy default config to `~/.ensemble_memory/config.toml`
4. Register Stop + SessionStart hooks in Claude Code's `~/.claude/hooks.json`

## What Gets Created

```
~/.ensemble_memory/
├── config.toml                 # Configuration (model, timeouts, etc.)
├── memory.db                   # SQLite database (structured memories)
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
| `ENSEMBLE_MEMORY_TIMEOUT` | `30` | Extraction timeout (seconds) |
| `ENSEMBLE_MEMORY_PYTHON` | `python3` | Python interpreter path |

## Features

### Capture (working now)
- Regex triage gate (<5ms) — only fires LLM on memory-worthy turns
- Ollama `format: "json"` constrained generation for reliable output
- SQLite hub with temporal metadata (decay rates, stability, supersession chains)
- Content-hash dedup (no duplicate memories)
- Content-similarity supersession (new facts replace old contradicting ones)
- Reinforcement tracking (repeated patterns flag promotion candidates)
- Human-readable markdown daily logs

### Recall (working now)
- SessionStart hook loads standing corrections and rules (importance >= 7)
- Injected as `additionalContext` into Claude Code's system prompt

### Planned (Phase 2)
- Semantic search via ONNX BGE-M3 embeddings + Milvus Lite
- Query-time retrieval (retrieve relevant memories when topic changes mid-session)
- Embedding-based supersession (cosine similarity replaces Jaccard)
- Nightly batch processor for missed memories

## Architecture

Built from a 6-round Delphi process with 6 expert AI agents, each specializing in a memory dimension:

1. **Semantic** — Embedding + BM25 hybrid search (Memsearch-inspired)
2. **Hybrid RAG** — Decision extraction + cross-encoder reranking (ClawMem-inspired)
3. **Contextual** — Chunk enrichment before embedding (Anthropic Contextual Retrieval)
4. **Temporal** — Recency decay, Ebbinghaus forgetting curves, ACT-R activation
5. **Knowledge Graph** — Entity-relationship graphs, A-MEM Zettelkasten
6. **Cognitive** — Memory triage, classification, forgetting policies

Full design: [final_design.md](../synthesis/final_design.md) (1,931 lines)

## Tests

```bash
python3 tests/test_ensemble_memory.py
```

45 tests covering triage, SQLite hub, markdown writer, session start, and end-to-end integration.

## Uninstall

```bash
# Remove hooks from Claude Code settings
# (manually edit ~/.claude/settings.json and remove Stop/SessionStart entries)

# Remove data
rm -rf ~/.ensemble_memory/
```

## License

MIT

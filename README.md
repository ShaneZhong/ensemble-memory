# Ensemble Memory System

A 100% free, local memory system for AI coding agents. Captures corrections, decisions, and insights from your sessions automatically — then retrieves them semantically when relevant. Zero API cost, runs entirely on your machine.

Supports **100+ languages** including Chinese, Japanese, and Korean via BGE-M3 multilingual embeddings.

## How It Works

```
Claude Code session
    |
    v
[SessionStart hook] ──> Load standing rules (importance >= 7) ──> Inject into context
    |                    + recent context (semantic/episodic)
    v
[User types a prompt]
    |
    v
[UserPromptSubmit hook] ──> Query daemon ──> Find similar memories ──> Inject relevant ones
    |                        (~50ms)          (RRF: cosine + BM25      via hookSpecificOutput
    |                                         + cross-encoder rerank)
    |                                              |
    |                                              v
    |                                   [KG neighborhood lookup]
    |                                   (keyword → FTS5 → BFS 2-hop)
    |                                   [Inject entity/relationship context]
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
                                      (extracts memories +
                                       entities + relationships)
                                              |
                                              v
                                   [SQLite + Markdown write]
                                   [KG entity/relationship upsert]
                                              |
                                              v
                                   [Daemon /embed + /invalidate_cache]
                                   [A-MEM evolution queue]
```

**Three hooks + one daemon:**
1. **SessionStart** — loads standing rules + recent context as baseline
2. **UserPromptSubmit** — queries the embedding daemon for semantically relevant memories
3. **Stop** — captures new corrections and decisions after each response
4. **Embedding daemon** — persistent background process (port 9876) that loads BGE-M3 (~600MB RAM), cross-encoder reranker, and serves embedding + search requests in ~50ms

## What Gets Captured

- Corrections: "no, don't use MySQL, use PostgreSQL"
- Decisions: "let's use SQLite for the database"
- Procedural rules: derived patterns from repeated corrections
- Entities: named technologies, tools, projects, people, rules extracted from each turn
- Relationships: typed edges between entities (e.g. `Python USES SQLite`, `Redis CONFLICTS_WITH MySQL`)

## What Doesn't Get Captured

- Ordinary requests: "read this file", "run the tests"
- Code output, tool results, assistant responses

## Requirements

- **Ollama** with `qwen2.5:3b` model (~1.9GB)
- **Python 3.10+**
- **sentence-transformers** (`pip install sentence-transformers`) — for BGE-M3 embeddings (~1.7GB first download)
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
| `ENSEMBLE_MEMORY_EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim, multilingual) |
| `ENSEMBLE_MEMORY_TIMEOUT` | `30` | Extraction timeout (seconds) |
| `ENSEMBLE_MEMORY_PYTHON` | `python3` | Python interpreter path |
| `ENSEMBLE_MEMORY_ENRICH_ENABLED` | `1` | Enable contextual enrichment (0 to disable) |
| `ENSEMBLE_MEMORY_MIN_ENRICH_IMPORTANCE` | `6` | Minimum importance for enrichment |
| `ENSEMBLE_MEMORY_CROSS_ENCODER` | `1` | Enable cross-encoder reranking (0 to disable) |
| `ENSEMBLE_MEMORY_COMPOSITE_SCORING` | `1` | Use weighted composite scoring (0 for legacy) |

## Features

### Capture
- Regex triage gate (<5ms) — only fires LLM on memory-worthy turns
- Ollama `format: "json"` constrained generation for reliable output
- SQLite hub with full temporal metadata (decay rates, stability, supersession chains)
- Content-hash dedup at SQLite level (no duplicate memories)
- Near-dedup via Jaccard similarity (>= 0.85) — catches rephrased duplicates
- Embedding generated on insert (BAAI/bge-m3, 1024-dim, 100+ languages)
- **Decision vault**: typed decision index (ARCHITECTURAL, PREFERENCE, ERROR_RESOLUTION, CONSTRAINT, PATTERN) with FTS5 BM25 search
- **Reinforcement tracking**: repeated procedural patterns increment count, boost stability, and promote to CLAUDE.md at 5+ reinforcements
- **Pipeline queue**: typed routing to expert processors with retry logic and error tracking
- Human-readable markdown daily logs as source of truth

### Recall
- **SessionStart**: loads standing corrections and rules (importance >= 7) + recent semantic/episodic context as baseline. Validity gates exclude expired/future memories.
- **UserPromptSubmit**: hybrid retrieval — RRF fusion of cosine similarity + BM25 keyword search + temporal decay + importance scoring. Optional cross-encoder reranking (top-20 → top-5). Injects only relevant memories as context BEFORE Claude responds.
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top RRF results for precision (opt-in per query)
- **Composite scoring**: weighted sum of RRF (0.5), temporal (0.3), importance (0.2) with confidence modulation
- Temporal decay scoring (ACT-R Petrov + Ebbinghaus forgetting curve) weights fresher memories higher
- **Importance decay**: memories not accessed in 30 days lose 1 importance point (floor at 3), keeping retrieval surface fresh

### Knowledge Graph
- **Entity extraction**: entities (name, type, description) extracted from each turn alongside memories
- **Entity dedup**: exact name match first, then FTS5 fuzzy lookup, then cosine similarity (> 0.85) to merge similar entities
- **Relationship edges**: typed predicates (USES, DEPENDS_ON, CONFLICTS_WITH, ...) with confidence scores
- **Predicate normalization**: LLM output like `"CAUSES | AFFECTS"` normalized to first valid predicate
- **2-hop BFS traversal**: bidirectional BFS via recursive SQLite CTE with cycle detection
- **FTS5 index**: full-text search on entity names, aliases, and descriptions
- **KG-enriched retrieval**: `/search` daemon endpoint extracts keywords, queries FTS5, runs BFS neighborhood, injects entity+relationship context alongside semantic memories
- **Cold-start bootstrap**: `bootstrap_from_files()` processes CLAUDE.md and MEMORY.md in 2000-char chunks, populating the KG from existing project docs
- **Decay config**: per-predicate decay windows (e.g. `HAS_VERSION` = 30 days, `APPLIES_TO` = permanent)
- **Episode tracking**: `kg_episodes` table records session turns; `kg_appears_in` links entities to episodes

### A-MEM Evolution
- **Memory relationship classification**: Ollama classifies semantic relationships between memories (SUPPORTS, REFINES, CONTRADICTS, SUPERSEDES, EVOLVED_FROM, ENABLES, CAUSED_BY, RELATED)
- **V2 prompt**: few-shot classification with strength calibration guidance
- **Async queue**: memories queued for evolution via pipeline queue, processed by daemon background jobs
- **Accuracy evaluation**: 50 ground-truth pairs covering all 8 link types

### Supersession
- **Structured**: same subject + predicate → old fact automatically superseded
- **Cosine similarity** (>= 0.85): semantically similar corrections auto-supersede
- **Jaccard fallback** (>= 0.6): word-overlap supersession when no embeddings available
- Superseded memories are retained for history but excluded from retrieval
- **Event bus**: trilateral sync — temporal, KG, and contextual experts each process supersession events independently
- **Chain depth pruning**: per-type limits (correction: 2, procedural: 3, semantic: 3, episodic: 5) prevent infinite chains

### Lifecycle Management
- **Reinforcement pipeline**: procedural memories tracked across sessions; stability increases at counts 2, 3, 5
- **Promotion to CLAUDE.md**: procedural rules reinforced 5+ times within 180 days auto-promote with file locking
- **Community detection**: NetworkX Louvain on entity-relationship graph (with SQLite CTE fallback)
- **Relationship decay**: per-predicate TTL windows (HAS_VERSION: 30d, USES: 180d, APPLIES_TO: permanent)
- **Garbage collection**: soft-delete via two-path eligibility (chain-pruned OR forgotten+superseded), protected memories (importance >= 9) never GC'd
- **Background jobs**: daemon runs serialized maintenance (temporal batch, event bus, chain pruning, GC, community detection, relationship decay, promotion, A-MEM evolution) every 6 hours

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
│   ├── embedding_daemon.py    # Persistent HTTP server (port 9876, ~600MB RAM)
│   │                          #   BGE-M3 embeddings + cross-encoder reranker
│   │                          #   endpoints: /search (RRF fusion + rerank),
│   │                          #   /embed, /embed_batch, /invalidate_cache,
│   │                          #   /health, /status, /log_feedback
│   └── daemon_ctl.sh          # start/stop/restart/status management
├── hooks/
│   ├── db.py                  # SQLite hub (schema, connection, DDL, re-exports)
│   ├── db_memory.py           # Memory CRUD, embedding, enrichment, supersession
│   ├── db_lifecycle.py        # Reinforcement, pipeline queue, chain pruning, GC
│   ├── db_decisions.py        # Decision vault CRUD + BM25 search
│   ├── kg.py                  # Knowledge graph (entity resolution, BFS, communities)
│   ├── evolution.py           # A-MEM relationship classification via Ollama
│   ├── promote.py             # CLAUDE.md promotion pipeline (file locking)
│   ├── enrich.py              # Contextual enrichment (KG prefix + LLM)
│   ├── embeddings.py          # Sentence-transformers wrapper (BAAI/bge-m3, 1024-dim)
│   ├── triage.py              # Regex signal detection (< 5ms)
│   ├── extract.py             # Ollama qwen2.5:3b caller with JSON validation
│   ├── write_log.py           # Markdown daily log writer with dedup
│   ├── store_memory.py        # SQLite + embedding + markdown orchestrator
│   ├── session_start.sh/py    # SessionStart hook — load standing rules + recent context
│   ├── stop.sh                # Stop hook — capture corrections + decisions
│   ├── user_prompt_submit.sh/py  # UserPromptSubmit hook — thin HTTP client to daemon
│   └── prompts/extraction.txt # LLM prompt template
├── scripts/
│   └── migrate_embeddings.py  # Re-embed all memories (for model upgrades)
├── tests/                     # 402 tests across 7 test files
├── sprints/                   # AutoShip sprint research + plans
├── docs/roadmap.md            # Phase roadmap + decision log
├── config/default_config.toml
├── install.sh
└── HOOKS_REFERENCE.md         # Hook payload/response format docs
```

## Tests

```bash
python3 -m pytest tests/ -v
```

402 tests, all passing:
- **Phase 1-5** (116 tests): triage, DB CRUD, temporal scoring, supersession, dedup, embeddings, query retrieval, knowledge graph, decision vault, contextual enrichment
- **Phase 5 Enrichment** (22 tests): KG path, LLM path, batch enrichment, store_memory integration
- **Phase 6 Lifecycle** (111 tests): reinforcement, promotion, supersession event bus, chain pruning, community detection, relationship decay, GC, background jobs, e2e integration
- **Phase 7 Recall + A-MEM** (46 tests): composite scoring, recall quality, A-MEM memory links, evolution queue
- **Phase 8 Cross-encoder + Calibration** (27 tests): cross-encoder reranking, truncation, A-MEM V2 prompt, SessionStart validity gates, recent context injection
- **Phase 8 A-MEM Eval** (30 tests): 50 ground-truth pairs, per-type accuracy, prompt format validation
- **Phase 9 Embedding Upgrade** (22 tests): BGE-M3 model config, truncation limits, re-embed migration, pipeline queue CRUD, A-MEM queue migration
- **Daemon** (19 tests): /embed, /embed_batch endpoints, error handling
- **Phase 4 Integration** (9 tests): decision vault + BM25 search integration

## Roadmap

- [x] Phase 1: Capture pipeline (regex triage → Ollama extraction → SQLite + markdown)
- [x] Phase 2: Semantic search (embedding daemon, cosine similarity, cosine supersession)
- [x] Phase 3: Knowledge graph (entity extraction, BFS traversal, FTS5, cold-start bootstrap)
- [x] Phase 4: Decision vault (typed decisions, BM25+RRF fusion, importance decay)
- [x] Phase 5: Contextual enrichment (KG/LLM prefix generation, enriched embeddings)
- [x] Phase 6: Lifecycle management (reinforcement, promotion, event bus, chain pruning, community detection, GC)
- [x] Phase 7: Recall quality + A-MEM evolution (composite scoring, memory relationship classification)
- [x] Phase 8: Cross-encoder reranking + A-MEM calibration (V2 prompt, validity gates, recent context)
- [x] Phase 9: Embedding upgrade (BGE-M3 1024-dim multilingual, pipeline queue table)
- [ ] Phase 9.2: Milvus Lite vector search (deferred until >1K memories)
- [ ] Phase 10: Hooks & capture completeness (PreCompact hook, ClawMem session tables)
- [ ] Phase 11: Monitoring & operations (health dashboard, memory browser, export/import)
- [ ] Phase 12: Multi-project federation (cross-project query, memory sharing)

See [docs/roadmap.md](docs/roadmap.md) for detailed task breakdown and decision log.

## Uninstall

```bash
# Remove hooks from Claude Code settings
# (edit ~/.claude/settings.json and remove Stop/SessionStart/UserPromptSubmit entries)

# Remove data
rm -rf ~/.ensemble_memory/
```

## License

MIT

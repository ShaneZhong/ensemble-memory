# Ensemble Memory System — Test Results & Known Issues

**Date**: 2026-04-01
**Phase**: Phase 9 Complete (Embedding Upgrade & Scale)
**Test Environment**: Mac Mini M4 16GB, Claude Code, Ollama qwen2.5:3b, sentence-transformers BAAI/bge-m3 (1024-dim)

---

## Test Results

### Automated Tests (402/402 PASS)

Run: `cd ensemble-memory && python3 -m pytest tests/ -v`

| Test File | Tests | Phase | Coverage |
|-----------|-------|-------|----------|
| test_ensemble_memory.py | 116 | 1-5 | Triage, DB, write log, session start, integration, embeddings, query retrieval, cosine supersession, knowledge graph, daemon endpoints, decision vault, contextual enrichment |
| test_phase5_enrichment.py | 22 | 5 | KG path, LLM path, batch enrichment, store_memory integration |
| test_phase6.py | 111 | 6 | Reinforcement, promotion, supersession event bus, chain pruning, community detection, relationship decay, GC, background jobs, e2e integration |
| test_phase7.py | 46 | 7 | Composite scoring, recall quality, A-MEM memory links, evolution queue |
| test_phase8.py | 27 | 8 | Cross-encoder reranking, truncation, SessionStart validity gates, recent context injection |
| test_amem_eval.py | 30 | 8 | 50 ground-truth pairs, per-type accuracy, prompt format validation |
| test_phase9.py | 22 | 9 | BGE-M3 model config, truncation limits, re-embed migration, pipeline queue CRUD, A-MEM queue migration |
| test_daemon.py | 19 | 2-3 | /embed, /embed_batch endpoints, error handling |
| test_phase4_integration.py | 9 | 4 | Decision vault + BM25 search integration |
| **Total** | **402** | | **All passing** |

### Historical Live Tests

These were run during early development and validated the core capture/retrieval loop.

<details>
<summary>Phase 1-3 Live Tests (2026-03-25 to 2026-03-28)</summary>

| Test | Input | Result | Notes |
|------|-------|--------|-------|
| Correction capture | "no, don't use system Python" | PASS | importance 7 |
| Decision capture | "let's use SQLite for the database" | PASS | importance 6 |
| No false positive | "can you read the file at progress.md" | PASS | Regex found no signals |
| SessionStart loading | Start new session | PASS | Context loaded via additionalContext |
| Capture + embed | "don't use print, use logging" | PASS | After embed-via-daemon fix |
| Retrieve correction | "how to add debug output" | PASS | Claude responded with logging module |
| Entity extraction | "don't use MySQL, use Redis" | PASS | Entities + relationships captured |
| Entity dedup | "use Redis Stack instead of plain Redis" | PASS | Redis entity merged |
| KG-enriched retrieval | "what cache for graph data?" | PASS | KG context injected |
| Cold-start bootstrap | bootstrap_from_files() | PASS | 28 entities, 27 relationships |

</details>

---

## Known Issues

### RESOLVED — `additionalContext` is weak

**Status**: Fixed in Phase 2. Query-time retrieval via embedding daemon injects relevant memories at the moment they're needed, not just at session start.

### LOW — Duplicate memories in markdown

**Problem**: The same memory can appear 2-3x in the daily markdown log when the LLM rephrases slightly.

**Mitigation**: Content-hash dedup at SQLite level prevents duplicate DB entries. Markdown logs may have near-duplicates but the retrieval system deduplicates.

### LOW — Sessions table sparse

**Problem**: `sessions` table is not consistently populated.

**Impact**: Minor — session tracking is supplementary to memory capture.

### INFO — Extraction latency

**Observation**: Ollama extraction takes 6-13 seconds per turn (average ~9s). Not user-facing (runs via Stop hook).

**Mitigation**: Regex triage gate (<5ms) ensures LLM only fires on memory-worthy turns. Pipeline queue provides retry for timeouts.

---

## How to Verify

```bash
# Run all tests
python3 -m pytest tests/ -v

# Check today's markdown log
cat ~/.ensemble_memory/memory/$(date +%Y-%m-%d).md

# Check SQLite memories
python3 -c "
import sqlite3, json
conn = sqlite3.connect('\$HOME/.ensemble_memory/memory.db')
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT substr(content,1,60) as content, memory_type, importance FROM memories WHERE superseded_by IS NULL AND gc_eligible = 0 ORDER BY created_at DESC LIMIT 10'):
    print(dict(r))
"

# Check embedding dimensions
python3 -c "
import sqlite3, json
conn = sqlite3.connect('\$HOME/.ensemble_memory/memory.db')
row = conn.execute('SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1').fetchone()
print(f'Embedding dim: {len(json.loads(row[0]))}')
"

# Check pipeline queue health
python3 -c "
import sys; sys.path.insert(0, 'hooks')
import db
print(db.get_pipeline_stats())
"

# Check extraction stats
cat ~/.ensemble_memory/logs/extraction_stats.jsonl | tail -10
```

---

## Next Steps

See [docs/roadmap.md](docs/roadmap.md) for the full roadmap. Key upcoming phases:

- **Phase 9.2**: Milvus Lite vector search (deferred until >1K memories)
- **Phase 10**: PreCompact hook, ClawMem session tables
- **Phase 11**: Health dashboard, memory browser, export/import
- **Phase 12**: Multi-project federation

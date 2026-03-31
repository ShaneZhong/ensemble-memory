# Sprint 4 (Phase 9.1) Research: BGE-M3 1024-dim Migration

## Objective
Swap embedding model from `all-MiniLM-L6-v2` (384-dim) to `BAAI/bge-m3` (1024-dim) for better multilingual support (especially Chinese) and longer context window.

## Current State

### Embedding Model Usage
1. **`hooks/embeddings.py`** — Shared embedding service
   - `MODEL_NAME = os.environ.get("ENSEMBLE_MEMORY_EMBED_MODEL", "all-MiniLM-L6-v2")`
   - `get_embedding()` returns 384-dim vector
   - `get_embeddings()` batch embed, batch_size=32
   - Used by: stop hook extraction, enrichment, A-MEM similarity

2. **`daemon/embedding_daemon.py`** — HTTP daemon
   - Loads `all-MiniLM-L6-v2` directly via SentenceTransformer
   - `_get_embedding()` encodes with `normalize_embeddings=True`
   - `/embed` endpoint truncates to 512 chars (MiniLM limit)
   - `/embed_batch` also truncates to 512 chars
   - `_find_similar_for_amem()` uses daemon's own embedding

3. **Storage**: `memories.embedding` column stores JSON-serialized `list[float]`
   - No dimension validation — just stores whatever the model produces
   - Cosine similarity computed in Python (no SQLite vec extension)

### Model Comparison
| Property | all-MiniLM-L6-v2 | BAAI/bge-m3 |
|----------|-------------------|-------------|
| Dimensions | 384 | 1024 |
| Model size | ~80MB | ~1.7GB |
| Max tokens | 256 | 8192 |
| Languages | English-focused | 100+ (strong Chinese) |
| RAM at load | ~200MB | ~600MB |
| Encode speed | ~5ms/query | ~8ms/query |

### Key Constraint
- Mac Mini M4 16GB RAM
- Daemon keeps model resident in memory
- Cross-encoder also loaded (~85MB)
- Total RAM budget: model (~600MB) + cross-encoder (~85MB) + cache + SQLite ≈ ~1GB, fine for 16GB

## Migration Requirements

1. **Model swap**: Change model name in both `embeddings.py` and `embedding_daemon.py`
2. **Truncation update**: MiniLM truncates at 512 chars; BGE-M3 handles 8192 tokens (~16K chars). Update `/embed` and `/embed_batch` truncation.
3. **Re-embedding**: All existing 384-dim embeddings must be re-generated at 1024-dim. Need a migration script.
4. **Dimension awareness**: Code should be dimension-agnostic (already is — cosine similarity works on any dim).
5. **Feature flag**: `ENSEMBLE_MEMORY_EMBED_MODEL` env var already exists for model override.
6. **Backward compat**: During migration, mixed-dimension embeddings will exist. Cosine similarity between 384-dim and 1024-dim vectors will error. Migration must be atomic or handle mixed dims.

## Risks
- **Download size**: ~1.7GB first-time download. Should happen at daemon startup, not during hook execution.
- **Memory**: ~600MB vs ~200MB. Acceptable on M4 16GB.
- **Migration time**: Re-embedding N memories at ~8ms each. 1000 memories ≈ 8 seconds. Fast enough for inline migration.
- **Mixed dimensions during migration**: Must clear old embeddings and re-embed in one batch to avoid mixed-dim cosine errors.

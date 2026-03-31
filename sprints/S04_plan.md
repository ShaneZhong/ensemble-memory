# Sprint 4 (Phase 9.1) Plan: BGE-M3 1024-dim Migration

## Scope
Swap embedding model from all-MiniLM-L6-v2 (384-dim) to BAAI/bge-m3 (1024-dim). Update all embedding code paths. Provide re-embedding migration. All existing memories get 1024-dim embeddings.

## Files Modified

### 1. `hooks/embeddings.py`
- Change default `MODEL_NAME` to `BAAI/bge-m3`
- Update docstring: 384-dim → 1024-dim, ~80MB → ~1.7GB
- Update `get_embedding()` docstring: "Returns 1024-dim vector"
- No code logic changes needed — SentenceTransformer API is model-agnostic

### 2. `daemon/embedding_daemon.py`
- Change `_load_model()`: `SentenceTransformer("BAAI/bge-m3")`
- Update `/embed` truncation: 512 → 8192 chars (BGE-M3 handles 8K tokens)
- Update `/embed_batch` truncation: 512 → 8192 chars
- Update docstring comment about truncation
- Add `EMBEDDING_DIM` constant (1024) for documentation, not enforcement

### 3. `hooks/db_memory.py` (new function)
- Add `reembed_all_memories()` migration function:
  - Fetches all memories with non-NULL embeddings
  - Batches them (batch_size=32)
  - Re-embeds using current model
  - Updates `memories.embedding` column
  - Returns count of re-embedded memories
  - Logs progress every 100 memories

### 4. `scripts/migrate_embeddings.py` (new file)
- Standalone migration script
- Imports embeddings module, calls reembed_all_memories()
- Can be run manually: `python scripts/migrate_embeddings.py`
- Prints progress and summary
- Idempotent — safe to re-run

## Acceptance Criteria
1. `embeddings.py` defaults to `BAAI/bge-m3` model
2. `embedding_daemon.py` loads `BAAI/bge-m3` and produces 1024-dim vectors
3. `/embed` and `/embed_batch` truncate at 8192 chars (not 512)
4. `reembed_all_memories()` function exists and works
5. Migration script is runnable standalone
6. All existing tests pass (dimension-agnostic)
7. New tests verify:
   - Model name is BAAI/bge-m3 by default
   - Embedding dimension is 1024
   - Truncation limits updated
   - `reembed_all_memories()` updates stored embeddings
   - Mixed-dimension protection (skip if model unavailable)

## Test Plan (tests/test_phase9.py)
- `TestBGEM3ModelConfig` (3 tests): model name, dimension constant, env override
- `TestEmbeddingTruncation` (2 tests): /embed and /embed_batch use 8192 limit
- `TestReembedMigration` (5 tests): batch re-embed, empty DB, error handling, idempotency, progress logging

## Risks & Mitigations
- **Model download**: First `SentenceTransformer("BAAI/bge-m3")` triggers ~1.7GB download. This happens at daemon startup, not during hooks. Acceptable.
- **RAM**: ~600MB for BGE-M3 vs ~200MB for MiniLM. Within M4 16GB budget.
- **Test isolation**: Tests mock the model to avoid requiring 1.7GB download in CI. Use `unittest.mock.patch` to substitute a fake model that returns vectors of correct dimension.

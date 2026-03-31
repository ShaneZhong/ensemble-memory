# Sprint 8.1 Research — Cross-encoder Reranking

## Relevant Existing Code
- `daemon/embedding_daemon.py:426-547` — `_search()` does cosine + BM25 → RRF fusion → top-5. No reranking step.
- `daemon/embedding_daemon.py:390-423` — `composite_score()` for weighted scoring.
- `daemon/embedding_daemon.py:62-82` — `_load_model()` loads `all-MiniLM-L6-v2` bi-encoder.
- Spec Section 6.2 — Cross-encoder ALWAYS for Stop hook, NEVER for UserPromptSubmit.

## Spec Requirements
- Deploy `ms-marco-MiniLM-L-6-v2` (~85MB) via `sentence-transformers` `CrossEncoder`.
- Rerank top-20 RRF results down to top-5 in Stop hook path.
- UserPromptSubmit has <5ms budget — must NOT use cross-encoder.
- For chunks >600 chars, pass heading + first 200 chars only.

## Patterns to Reuse
- `_load_model()` pattern for lazy loading cross-encoder model.
- `_search()` already has RRF fusion — add reranking as step between fusion and final return.
- Feature flag pattern: `ENSEMBLE_MEMORY_CROSS_ENCODER` env var (default "1").

## Risks
- Cross-encoder adds ~85MB RAM + ~200ms per rerank call. Must coexist with Ollama on 16GB.
- `sentence-transformers` may not have `CrossEncoder` class in all versions — need to handle ImportError.
- If cross-encoder loading fails, must degrade gracefully to RRF-only (no crash).

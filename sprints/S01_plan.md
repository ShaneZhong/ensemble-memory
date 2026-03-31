# Sprint 8.1 Plan — Cross-encoder Reranking (REVISED)

## Goal
Add cross-encoder reranking to the search pipeline, reranking top-20 RRF results to top-5 for improved recall precision. Feature-flagged and gracefully degrading.

## Changes
| File | Change | Est. LOC |
|------|--------|----------|
| daemon/embedding_daemon.py | Add `_load_cross_encoder()` with thread lock, `_cross_encoder_rerank()`, update `_search()` to accept `rerank` param, update `do_POST /search` handler to pass `rerank` from body | ~80 |
| tests/test_phase8.py | Unit + integration tests for cross-encoder reranking | ~130 |

## Design Decisions

### Rerank default: `false` (caller opts IN)
Per spec: cross-encoder is "ALWAYS for Stop hook, NEVER for UserPromptSubmit." Default `rerank=false` in `/search` endpoint. Stop hook callers explicitly pass `rerank: true`. This puts the safety on the right side — no accidental expensive reranking.

### Score integration: pure reorder
Cross-encoder score replaces `final_score` for the reranked candidates. Rationale: the cross-encoder is a precision tool — its whole purpose is to reorder after RRF fusion. Blending would dilute its signal. After reranking, the top-5 results use cross-encoder score as `final_score`.

### Content truncation
For content >600 chars: `subject + ": " + content[:200]` if `subject` is non-empty, else `content[:200]`. The `subject` field (from triple extraction) serves as the "heading". This avoids adding a new field.

### Thread safety
Lazy-load cross-encoder behind `threading.Lock()`. First `/search` with `rerank=true` triggers load. Subsequent calls reuse the loaded model.

## Acceptance Criteria
- [ ] Cross-encoder model loaded lazily on first rerank request (not at startup)
- [ ] Thread-safe lazy loading via `threading.Lock()`
- [ ] Top-20 RRF candidates reranked to top-5 using cross-encoder scores
- [ ] Cross-encoder score replaces `final_score` (pure reorder, no blending)
- [ ] For content >600 chars, truncated to `subject: content[:200]`
- [ ] `rerank` param defaults to `false` — Stop hook callers opt in
- [ ] Feature flag `ENSEMBLE_MEMORY_CROSS_ENCODER` (default "1") master kill-switch
- [ ] Graceful degradation: if cross-encoder unavailable, falls back to RRF-only
- [ ] Latency target: <300ms per rerank of 20 candidates

## Test Plan
- Unit: `_cross_encoder_rerank()` with mocked model, various input sizes
- Unit: Content truncation — short content, long content, with/without subject
- Unit: Feature flag disable → no reranking
- Unit: Model load failure → graceful fallback
- Unit: `rerank=false` (default) → no cross-encoder invoked
- Unit: `rerank=true` → cross-encoder invoked
- Integration: Full search pipeline with reranking enabled
- Edge: Empty candidates, single candidate, all same scores
- Negative: Verify UserPromptSubmit path does NOT invoke cross-encoder

## Risks
- RAM pressure (~85MB for cross-encoder + ~90MB for bi-encoder). Mitigation: lazy loading, model only loaded when first rerank requested.
- `CrossEncoder` import may fail. Mitigation: try/except with `_has_cross_encoder` flag, degrades to RRF-only.
- Thread contention during lazy load. Mitigation: `threading.Lock()` ensures single load.

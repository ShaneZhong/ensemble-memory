# Sprint E1.1 — LongMemEval Provider Adapter (Plan)

**Phase**: E1 (Evaluation)
**Sprint**: 6 (E1.1)
**Date**: 2026-04-01

## Goal

Build a self-contained evaluation harness that ingests LongMemEval data, queries our ensemble memory retrieval, generates answers via Ollama, and scores them with an LLM judge.

## Changes

| File | Change | Est. LOC |
|------|--------|----------|
| `eval/longmemeval_adapter.py` | Provider adapter: ingest conversations → SQLite, query via search | ~150 |
| `eval/longmemeval_runner.py` | Runner: load dataset, run questions, generate answers, judge, report | ~200 |
| `eval/longmemeval_judge.py` | LLM-as-judge scoring adapted from LongMemEval prompts | ~80 |
| `eval/README.md` | Usage instructions | ~30 |
| `tests/test_longmemeval.py` | Tests for adapter + runner | ~100 |

## Design Decisions

### Architecture
```
longmemeval_oracle.json
        |
        v
[longmemeval_adapter.py]
  - Parse sessions → episodic memories
  - Batch embed via embeddings module
  - Store in isolated SQLite DB (not production DB)
        |
        v
[longmemeval_runner.py]
  - For each question:
    1. Query our search engine with question text
    2. Build context from retrieved memories
    3. Send question + context to Ollama (qwen2.5:3b)
    4. Collect hypothesis answer
        |
        v
[longmemeval_judge.py]
  - Compare hypothesis vs ground-truth answer
  - Use task-specific judge prompts (from LongMemEval)
  - Output: per-question pass/fail
        |
        v
  Report: per-ability accuracy, overall accuracy
```

### Isolated DB
Eval uses a separate SQLite DB (`/tmp/longmemeval_eval.db`) to avoid contaminating production memory. Set `db._DB_PATH_OVERRIDE` during ingestion/retrieval.

### Ingestion Strategy
Each conversation turn becomes an episodic memory:
- `content`: turn content (user or assistant message)
- `memory_type`: "episodic"
- `importance`: 5 (default) — turns with `has_answer: true` get importance 8
- `session_id`: from `haystack_session_ids`
- `created_at`: parsed from `haystack_dates`
- Embedding: batch-generated via BGE-M3

### Retrieval
Import `_search()` directly from daemon module. Override project filter to match eval DB. Return top-5 hits as context for the generator LLM.

### Judge Prompts
Adapted from LongMemEval `evaluate_qa.py`:
- **Default**: "Does the response contain the correct answer?"
- **Temporal**: "Allow off-by-one errors for dates/times"
- **Knowledge update**: "The correct answer reflects the LATEST information"
- **Abstention**: "The model should refuse/say it doesn't know"

## Acceptance Criteria

- [x] Adapter can ingest LongMemEval oracle dataset into isolated SQLite
- [x] All conversation turns embedded with BGE-M3
- [x] Runner queries our search engine and generates answers via Ollama
- [x] Judge scores each answer as correct/incorrect
- [x] Per-ability accuracy reported (5 abilities + abstention)
- [x] Overall accuracy reported
- [x] Latency per question reported (retrieval + generation)
- [x] Tests pass for adapter ingestion, retrieval, and judge logic (23 tests, 425 total)

## Test Plan

### Unit Tests
- `test_parse_sessions`: Verify conversation parsing extracts correct memories
- `test_ingestion_isolated_db`: Verify memories stored in temp DB, not production
- `test_retrieval_returns_results`: Verify search returns relevant hits
- `test_judge_correct_answer`: Verify judge scores correct answer as 1
- `test_judge_wrong_answer`: Verify judge scores wrong answer as 0
- `test_judge_abstention`: Verify abstention scoring works

### Integration Tests
- `test_ingest_and_retrieve`: Full pipeline from JSON → SQLite → search → results
- `test_small_eval_run`: Run 5 questions end-to-end, verify output format

### Edge Cases
- Empty sessions (no turns)
- Questions with multiple answer sessions
- Very long conversation turns (>8K tokens)
- Abstention questions (answer doesn't exist in history)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset download requires HuggingFace auth | Blocks eval | Direct URL download fallback |
| Ollama timeout on long contexts | Missing answers | Increase timeout, truncate context |
| Judge model hallucination | Inaccurate scoring | Compare sample against manual labels |

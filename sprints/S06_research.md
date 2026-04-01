# Sprint E1.1 — LongMemEval Provider Adapter (Research)

**Phase**: E1 (Evaluation)
**Sprint**: 6 (E1.1)
**Date**: 2026-04-01

## Goal

Build a provider adapter that ingests LongMemEval conversations into our ensemble memory system and answers benchmark questions using our retrieval pipeline.

## Relevant Existing Code

### Data Ingestion Path
- `hooks/store_memory.py:56-195` — `_store_to_sqlite()` handles memory insertion with supersession, reinforcement, enrichment
- `hooks/db.py:40-76` — DDL schema for `memories` table
- `hooks/db_memory.py:50-88` — Temporal scoring functions
- `hooks/kg.py` — Entity upsert, relationship insertion, BFS traversal

### Retrieval Path
- `daemon/embedding_daemon.py:503-636` — `_search()` function: RRF fusion of cosine + BM25 + cross-encoder reranking
- `daemon/embedding_daemon.py:467-500` — `composite_score()` weighted formula
- `daemon/embedding_daemon.py:292-356` — `_get_kg_context()` KG neighborhood lookup

### LongMemEval Data Format
```json
{
  "question_id": "q_001",
  "question_type": "single-session-user|multi-session|knowledge-update|temporal-reasoning|*_abs",
  "question": "What is the user's favorite restaurant?",
  "answer": "The user's favorite restaurant is Olive Garden",
  "question_date": "2024/05/15 (Wed) 14:00",
  "haystack_session_ids": ["session_1", "session_2"],
  "haystack_dates": ["2024/01/10 10:00", "2024/03/20 15:00"],
  "haystack_sessions": [
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "...", "has_answer": true}
    ]
  ],
  "answer_session_ids": ["session_1"]
}
```

### Evaluation Output Format
```jsonl
{"question_id": "q_001", "hypothesis": "model's answer here"}
```

## Spec Requirements

From `docs/evaluation-plan.md`:
1. Ingest LongMemEval conversations into our SQLite + KG
2. Route retrieval queries through our embedding daemon `/search`
3. Return context for LLM-as-Judge evaluation
4. Report per-ability accuracy, Recall@K, latency

## Key Design Decisions

### D1: Ingestion Strategy
LongMemEval conversations are multi-turn dialogues. We need to extract "memories" from them the same way our Stop hook does — via triage + LLM extraction. BUT running Ollama extraction on 500+ sessions would take hours.

**Decision**: Skip LLM extraction. Instead, directly insert conversation turns as episodic memories with structured metadata. This tests our RETRIEVAL quality, not our EXTRACTION quality. The benchmark questions already have ground-truth answers — we just need to find the right context.

### D2: Retrieval Interface
Two options:
- A) Call daemon HTTP `/search` endpoint (requires running daemon)
- B) Import `_search()` directly from daemon module (no HTTP overhead)

**Decision**: Option B — import directly. Avoids daemon dependency for benchmarking. The search logic is the same.

### D3: LLM Judge
LongMemEval uses GPT-4o as judge. We want free/local.

**Decision**: Use Ollama with qwen2.5:3b as judge. Adapt the evaluation prompts from LongMemEval's `evaluate_qa.py`. If judge quality is poor, we can upgrade to a larger model later.

### D4: Dataset Size
LongMemEval has 3 variants:
- `longmemeval_s` — ~40 sessions, 115K tokens (smallest)
- `longmemeval_m` — ~500 sessions per instance
- `longmemeval_oracle` — Ground-truth evidence sessions only

**Decision**: Start with `longmemeval_oracle` (oracle retrieval = ground-truth sessions). This tests whether our system can find the right answer when given the correct sessions. Then graduate to `longmemeval_s` to test retrieval quality.

## Patterns to Reuse

- `db.insert_memory()` for bulk ingestion
- `embeddings.get_embedding()` / `get_embeddings()` for batch embedding
- `_search()` logic from daemon for retrieval
- `db_memory.temporal_score()` for temporal scoring
- Existing test patterns from `test_phase9.py` for new eval tests

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| HuggingFace dataset download fails | Blocks eval | Cache locally, provide fallback URL |
| Ollama judge quality too low | Bad eval metrics | Compare subset against manual labels, upgrade model if needed |
| BGE-M3 embedding time for 500+ sessions | Slow ingestion | Batch embedding, progress bar |
| Memory usage with large conversation sets | OOM on M4 16GB | Process sessions incrementally, don't load all at once |

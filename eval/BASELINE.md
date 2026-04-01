# LongMemEval Baseline Results

**Date**: 2026-04-01
**Dataset**: longmemeval_oracle.json (500 questions, 10,866 memories, 940 sessions)
**System**: Ensemble Memory v9 (BGE-M3 1024-dim, RRF fusion, no cross-encoder reranking)
**Generator**: Ollama qwen2.5:3b
**Judge**: Ollama qwen2.5:3b
**Hardware**: Mac Mini M4 16GB

---

## Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **5.0% (25/500)** |
| Avg Retrieval Latency | 189ms |
| Avg Generation Latency | 1,267ms |
| Avg Judge Latency | 341ms |
| Total Eval Time | ~65 min (50 min embed + 15 min eval) |

### Per-Ability Breakdown

| Ability | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| Temporal Reasoning | 11 | 133 | 8.3% |
| Multi-Session Reasoning | 8 | 133 | 6.0% |
| Information Extraction | 6 | 156 | 3.9% |
| Knowledge Updates | 0 | 78 | 0.0% |
| Abstention | — | — | — (not in non-stratified run) |

### Retrieval Quality (20-question sample with context)

| Metric | Value |
|--------|-------|
| Hit rate | 100% (all questions got 5 hits) |
| Answer in context | 0% (retrieved wrong memories) |
| Root cause | Cosine similarity retrieves semantically similar but irrelevant turns |

---

## Configuration

```
Embedding model: BAAI/bge-m3 (1024-dim)
Search: RRF fusion (cosine + BM25 keyword)
Cross-encoder: disabled
Temporal filtering: disabled (benchmark memories from 2023)
Top-K: 5
Context format: turn-level (individual conversation turns as memories)
Ingestion: direct insert (no LLM extraction, no KG)
```

---

## Key Findings

1. **Retrieval precision is the primary bottleneck** — search returns 5 hits for every question but they are the wrong memories. Generic conversational turns with topic overlap outrank answer-bearing turns.

2. **Turn-level granularity loses session context** — each turn is an independent memory. A question about "what restaurant" retrieves turns mentioning "restaurant" from any session, not the specific session where the user discussed their favorite restaurant.

3. **qwen2.5:3b is too small for reasoning** — even when retrieval provides relevant context, the 3B model often responds "I don't have that information" or hallucinates.

4. **Knowledge Updates = 0%** — no mechanism to identify which of two contradictory memories is newer when both come from the same ingested dataset. Our supersession system works on live sessions (structured triples, cosine similarity) but not on bulk-ingested historical data.

5. **No abstention capability** — the system has no explicit abstention mechanism. It either finds memories or doesn't, but never says "this information wasn't discussed."

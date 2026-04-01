# LongMemEval Evaluation Harness

Benchmark the ensemble memory system against the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark — 500 questions across 5 memory abilities.

## Quick Start

```bash
# 1. Download the dataset (one-time)
cd eval/
curl -L -o longmemeval_oracle.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"

# 2. Run a quick 50-question eval (stratified across all abilities)
python3 longmemeval_runner.py longmemeval_oracle.json --limit 50

# 3. Run the full 500-question eval
python3 longmemeval_runner.py longmemeval_oracle.json --output results_full.json
```

## What It Tests

| Ability | Questions | What It Measures |
|---------|-----------|------------------|
| Information Extraction | 156 | Retrieving facts from past sessions |
| Multi-Session Reasoning | 133 | Synthesizing across conversations |
| Knowledge Updates | 78 | Handling superseded/contradictory info |
| Temporal Reasoning | 133 | Time-based relationships |
| Abstention | ~30 | Knowing when you can't answer |

## How It Works

```
longmemeval_oracle.json
       |
       v
[Adapter] Parse sessions → episodic memories → isolated SQLite
       |
       v
[Embedder] Batch BGE-M3 embedding (1024-dim)
       |
       v
[Runner] For each question:
  1. Query our RRF fusion search (cosine + BM25)
  2. Format top-5 hits as context
  3. Generate answer via Ollama (qwen2.5:3b)
       |
       v
[Judge] Ollama scores hypothesis vs ground truth
       |
       v
Report: per-ability accuracy, latency
```

## Options

```
--limit N        Run only N questions (stratified sample across abilities)
--skip-ingest    Skip ingestion if eval DB already exists
--db-path PATH   Custom DB path (default: /tmp/longmemeval_eval.db)
--output PATH    Save results JSON to path
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LONGMEMEVAL_GEN_MODEL` | `qwen2.5:3b` | Ollama model for answer generation |
| `LONGMEMEVAL_JUDGE_MODEL` | `qwen2.5:3b` | Ollama model for judging |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |

## Files

| File | Purpose |
|------|---------|
| `longmemeval_adapter.py` | Ingest conversations → SQLite + embeddings |
| `longmemeval_runner.py` | Run evaluation pipeline |
| `longmemeval_judge.py` | LLM-as-judge scoring |
| `longmemeval_oracle.json` | Dataset (download separately) |

## Notes

- Uses an isolated SQLite DB (`/tmp/longmemeval_eval.db`) — never touches production memory
- Temporal score filtering is disabled during eval (benchmark memories are from 2023)
- With `--limit`, questions are stratified-sampled across abilities for balanced coverage
- Judge accuracy with qwen2.5:3b is approximate — upgrade to a larger model for precise scoring

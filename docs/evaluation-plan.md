# Ensemble Memory System — Evaluation Plan

**Date**: 2026-04-01
**Status**: Planning
**Goal**: Benchmark our ensemble memory system against established memory evaluation frameworks.

---

## Landscape: AI Agent Memory Benchmarks (2025-2026)

### Tier 1 — Most Relevant to Our System

| Benchmark | Source | Focus | Repo |
|-----------|--------|-------|------|
| **LongMemEval** | Wu et al., ICLR 2025 | 500 questions across 5 memory abilities, scalable from 40 to 500+ sessions (115K-1.5M tokens) | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) |
| **Supermemory MemoryBench** | Supermemory, 2025 | Pluggable harness comparing providers (Mem0, Zep, Supermemory) with composite MemScore = accuracy / latency / context tokens | [supermemoryai/memorybench](https://github.com/supermemoryai/memorybench) |
| **AMA-Bench** | arXiv 2602.22769, Feb 2026 | Long-horizon agentic memory across 6 domains (Web, SQL, SWE, Gaming, Embodied AI) | Paper only |

### Tier 2 — Useful for Specific Components

| Benchmark | Source | Focus | Repo |
|-----------|--------|-------|------|
| **MemBench** | ACL Findings 2025 | Factual + reflective memory, effectiveness/efficiency/capacity | [import-myself/Membench](https://github.com/import-myself/Membench) |
| **AMemGym** | ICLR 2026 | Interactive on-policy benchmark, exposes "Reuse Bias" in static evals | [HuggingFace: AGI-Eval/AMemGym](https://huggingface.co/datasets/AGI-Eval/AMemGym) |
| **LoCoMo** | Snap Research, 2024 | 10 long conversations (300 turns, 35 sessions each), QA + summarization | [snap-research/locomo](https://github.com/snap-research/locomo) |
| **CloneMem** | arXiv 2601.07023 | AI clone long-term memory from digital traces | [AvatarMemory/CloneMemBench](https://github.com/AvatarMemory/CloneMemBench) |

### Tier 3 — Reference / Survey

| Resource | URL |
|----------|-----|
| Agent Memory Paper List | [Shichun-Liu/Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) |
| Awesome Memory for Agents | [TsinghuaC3I/Awesome-Memory-for-Agents](https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents) |
| Awesome Agent Memory | [TeleAI-UAGI/Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) |
| Letta Blog: Benchmarking AI Agent Memory | [letta.com/blog/benchmarking-ai-agent-memory](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| Memory in the Age of AI Agents (survey) | [arXiv 2512.13564](https://arxiv.org/abs/2512.13564) |

---

## LongMemEval — 5 Memory Abilities

LongMemEval is the gold standard. It tests 5 core abilities:

1. **Information Extraction** — retrieving specific facts from past sessions
2. **Multi-Session Reasoning** — synthesizing across conversations
3. **Knowledge Updates** — handling superseded/contradictory info (88 questions)
4. **Temporal Reasoning** — time-based relationships (76 questions)
5. **Abstention** — knowing when you cannot answer

### Mapping to Our Components

| LongMemEval Ability | Our Component | Expected Strength |
|---------------------|--------------|-------------------|
| Information Extraction | BGE-M3 + BM25 RRF fusion | Strong — 1024-dim multilingual embeddings |
| Multi-Session Reasoning | KG 2-hop BFS + A-MEM evolution | Strong — entity graph connects sessions |
| Knowledge Updates | Supersession (cosine + Jaccard + structured) | Very strong — trilateral event bus, chain pruning |
| Temporal Reasoning | Ebbinghaus + ACT-R decay model | Very strong — per-type decay rates, stability |
| Abstention | Importance threshold + temporal score floor | Moderate — implicit via score thresholds |

---

## Recommendation: Phased Evaluation

### Phase E1: LongMemEval (recommended first)

**Why**: Directly tests our two differentiators (supersession + temporal decay) with 500 pre-built questions and ground truth.

**Approach**:
1. Clone [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
2. Write a provider adapter that:
   - Ingests LongMemEval conversations into our SQLite + KG
   - Routes retrieval queries through our embedding daemon `/search`
   - Returns context for LLM-as-Judge evaluation
3. Run the 500-question eval using Ollama as the judge (free, local)
4. Report per-ability accuracy, Recall@K, latency

**Estimated effort**: ~200 LOC adapter + runner script

### Phase E2: Ablation Study

**Why**: Quantify each component's contribution.

**Approach**: Disable one component at a time on LongMemEval and measure accuracy drop:
- Baseline (all components)
- No cross-encoder reranking
- No KG context
- No temporal decay (all memories equal recency)
- No A-MEM evolution
- BM25-only (no embeddings)
- Embeddings-only (no BM25)

### Phase E3: Supermemory MemoryBench Harness

**Why**: Apples-to-apples comparison against Mem0, Zep, Supermemory.

**Approach**:
1. Write a TypeScript provider adapter (ingest, index, search interface)
2. Run against LoCoMo + LongMemEval + ConvoMem datasets
3. Report composite MemScore (accuracy / latency / context tokens)

### Phase E4: Custom Supersession Benchmark

**Why**: No existing benchmark has enough supersession tests. This is our strongest feature.

**Approach**:
1. Expand existing 50 ground-truth pairs (Phase 8) to 200+
2. Categories: structured supersession, cosine supersession, Jaccard fallback, chain pruning, false negatives (similar but NOT superseding)
3. Report precision/recall/F1 per supersession type

---

## Gap Analysis

| What We Do | Benchmark Coverage | Gap |
|------------|-------------------|-----|
| Semantic search (BGE-M3 + BM25 RRF) | Well covered by LongMemEval, MemBench | None |
| Knowledge graph (2-hop BFS) | Partially covered by MEMTRACK | Need custom multi-hop QA |
| Temporal decay (Ebbinghaus + ACT-R) | LongMemEval temporal reasoning (76 Qs) | Adequate |
| Decision vault | No benchmark exists | Need custom eval |
| Cross-encoder reranking | Measured via ablation | Adequate |
| A-MEM evolution | MemBench reflective scenarios | Partially covered |
| Supersession detection | LongMemEval knowledge updates (88 Qs) | Need expanded custom set |
| Pipeline queue | Operational, not accuracy | Latency benchmarks only |
| Multilingual (Chinese) | No benchmark tests this | Need custom CJK test set |

---

## Decision

**Start with Phase E1 (LongMemEval)**. It gives us the highest signal with the least effort — 500 pre-built questions with ground truth, directly testing supersession and temporal reasoning. All local, all free (Ollama as judge).

After E1, decide whether to pursue E2 (ablation), E3 (competitive comparison), or E4 (custom supersession) based on results.

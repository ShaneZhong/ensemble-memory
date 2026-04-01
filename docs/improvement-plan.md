# Ensemble Memory System — LongMemEval Improvement Plan

**Date**: 2026-04-01
**Baseline**: 5.0% (25/500) with qwen2.5:3b
**Target**: 75%+ (competitive with Zep/TiMem tier)
**Stretch**: 83%+ (competitive with Hindsight OSS)

---

## Competitive Landscape

| System | Model | Accuracy | Key Innovation |
|--------|-------|----------|----------------|
| Observational Memory | GPT-5-mini | 94.9% | Observer + Reflector agents, stable context window |
| Hindsight (TEMPR+CARA) | Gemini-3-pro | 91.4% | 4-way hybrid retrieval + disposition-aware reasoning |
| Hindsight (TEMPR+CARA) | OSS-20B | 83.6% | Same architecture, single consumer GPU |
| EmergenceMem | GPT-4o | 86.0% | Fact-augmented keys, session retrieval, CoT |
| Oracle GPT-4o | GPT-4o | 82.4% | Given correct sessions directly |
| Supermemory | GPT-4o | 81.6% | Atomic memories, relational versioning |
| TiMem | GPT-4o-mini | 76.9% | Temporal memory tree, complexity-aware recall |
| Zep/Graphiti | GPT-4o | 71.2% | Knowledge graph + temporal model |
| Full-context GPT-4o | GPT-4o | 60-64% | All 115K tokens, no retrieval |
| Naive RAG | Various | 52.0% | Simple turn-level retrieval |
| **Our baseline** | **qwen2.5:3b** | **5.0%** | **RRF fusion, turn-level, 3B model** |

---

## Gap Analysis: Why We Score 5% vs 52% (Naive RAG)

Our system scores **below** even naive RAG. Three compounding failures:

```
FAILURE CASCADE
═══════════════

1. RETRIEVAL PRECISION (biggest impact)
   ├── Turn-level granularity → loses session context
   ├── No index expansion → queries match topic words, not answer-bearing content
   ├── 10,866 memories = high noise → answer turns drowned out
   └── Impact: retrieves 5 hits but 0% contain the answer

2. GENERATOR MODEL (second biggest)
   ├── qwen2.5:3b can't reason over multi-turn context
   ├── Often says "I don't have that information" even with relevant context
   ├── Can't do temporal reasoning, multi-hop synthesis
   └── Impact: even with perfect retrieval, ~30-40% accuracy ceiling

3. MISSING CAPABILITIES
   ├── No temporal query expansion (time-aware search)
   ├── No knowledge update tracking (which fact is newer?)
   ├── No abstention mechanism
   └── No chain-of-thought prompting
```

---

## Improvement Phases (Ordered by Expected Impact)

### Phase I1: Session-Level Retrieval + Index Expansion
**Expected lift**: 5% → 25-35%
**Effort**: ~300 LOC
**Why first**: Biggest single improvement. Top systems all use session-level retrieval with expanded indices.

#### I1.1: Session-Level Memory Ingestion
Instead of storing each turn as a separate memory, store entire sessions as units:
```
CURRENT (turn-level):
  Memory 1: "[user] I love Italian food"
  Memory 2: "[assistant] What's your favorite restaurant?"
  Memory 3: "[user] Olive Garden, definitely"

IMPROVED (session-level):
  Memory 1: "Session about food preferences. User loves Italian food,
             favorite restaurant is Olive Garden."
  + Individual turns stored as sub-memories linked to session
```

**Implementation**:
- Group turns by `session_id` during ingestion
- Generate session summary using Ollama (qwen2.5:3b or 7b)
- Store both session summary (for retrieval) and individual turns (for context)
- Retrieve at session level, present turns as context

#### I1.2: Fact-Augmented Index Expansion
Extract structured facts from each session to augment search indices:
```
Session turns → Extract:
  - User facts: "favorite restaurant = Olive Garden"
  - Keyphrases: "Italian food, Olive Garden, dining preferences"
  - Temporal events: "2023-01-15: discussed restaurant preferences"
```

**Implementation**:
- Run Ollama extraction on each session during ingestion
- Store expanded keys alongside session embeddings
- Search matches on expanded keys, not just raw content

#### I1.3: Retrieve Sessions, Present Turns
When a question asks "What is the user's favorite restaurant?":
1. Search expanded keys → find "favorite restaurant = Olive Garden" session
2. Retrieve the full session turns as context
3. Present to generator with session structure preserved

---

### Phase I2: Upgrade Generator Model
**Expected lift**: 25-35% → 45-55%
**Effort**: ~50 LOC (config change + prompt engineering)
**Why second**: Once retrieval finds the right context, a better model can reason over it.

#### I2.1: Model Upgrade Options

| Model | Size | RAM | Speed | Expected Accuracy |
|-------|------|-----|-------|-------------------|
| qwen2.5:3b (current) | 1.9GB | ~3GB | ~1s | 5% (baseline) |
| qwen2.5:7b | 4.4GB | ~6GB | ~2s | ~30-40% |
| qwen2.5:14b | 8.7GB | ~10GB | ~4s | ~45-55% |
| llama3.1:8b | 4.7GB | ~6GB | ~2s | ~35-45% |
| gemma2:9b | 5.4GB | ~7GB | ~3s | ~35-45% |

**Recommendation**: Start with `qwen2.5:7b` (fits in 16GB alongside BGE-M3). If accuracy is still low, try `qwen2.5:14b` (tight on RAM but feasible).

#### I2.2: Chain-of-Thought Prompting
LongMemEval paper shows CoT adds **8-10 points**:
```
CURRENT PROMPT:
  "Based on the conversation history, answer the question."

IMPROVED PROMPT:
  "Based on the conversation history, answer the question.
   Think step by step:
   1. What relevant information is in the history?
   2. Does any information contradict or update earlier info?
   3. What is the most recent/relevant answer?
   Answer:"
```

#### I2.3: Separate Judge Model
Use a different (ideally larger) model for judging to reduce bias:
- Generator: qwen2.5:7b
- Judge: qwen2.5:14b or llama3.1:8b (independent model)

---

### Phase I3: Temporal Query Expansion
**Expected lift**: +7-11% on temporal questions (133 questions = ~10-15 more correct)
**Effort**: ~150 LOC
**Why third**: Temporal reasoning is our second-best ability but still only 8.3%.

#### I3.1: Time-Aware Query Parsing
Detect temporal signals in questions and expand queries:
```
Q: "What was the FIRST issue I had with my car AFTER its first service?"

Temporal signals: "first", "after"
→ Extract time constraints: event ordering required
→ Expand query: search for "car issue" AND "car service"
→ Order results by timestamp
→ Return the earliest "car issue" that occurs after "car service"
```

#### I3.2: Temporal Index on Sessions
Add temporal metadata to session summaries:
```
Session summary: "Discussed car problems after service"
Temporal index: {
  "date": "2023-02-15",
  "events": ["car service", "GPS malfunction"],
  "event_order": ["service" → "GPS issue"]
}
```

#### I3.3: Temporal Reasoning Prompt
For temporal questions, use a specialized prompt:
```
"The following conversations are ordered by date.
 Pay attention to the dates and order of events.
 Question: {question}

 Conversations (chronological order):
 [2023-01-10] Session about car purchase...
 [2023-02-15] Session about car service and GPS issues...

 Think about the timeline, then answer:"
```

---

### Phase I4: Knowledge Update Detection
**Expected lift**: +10-15% on knowledge update questions (78 questions)
**Effort**: ~200 LOC
**Why fourth**: Currently 0% on this ability — any improvement is pure gain.

#### I4.1: Fact Versioning During Ingestion
Track when facts change across sessions:
```
Session 1 (Jan): "My best 5K time is 28 minutes"
Session 3 (Mar): "I ran a 25:50 in the charity run!"

→ Fact version chain:
  best_5k_time: 28min (Jan) → 25:50 (Mar, LATEST)
```

#### I4.2: Latest-Fact Retrieval
When a question asks about a fact that has versions:
1. Retrieve all versions
2. Present them chronologically to the generator
3. Prompt: "The user's information may have changed over time. Use the MOST RECENT information."

#### I4.3: Supersession Chain in Context
Leverage our existing supersession system:
```
Context for generator:
  "Note: The user's 5K time was updated:
   - Initially: 28 minutes (Jan 2023)
   - Updated to: 25:50 (Mar 2023) ← LATEST

   Use the latest value."
```

---

### Phase I5: Cross-Encoder Reranking
**Expected lift**: +3-5% overall
**Effort**: ~30 LOC (already built, just needs enabling)
**Why fifth**: We already have cross-encoder (`ms-marco-MiniLM-L-6-v2`). Just needs to be enabled in the eval pipeline.

#### I5.1: Enable Reranking in Eval
```python
# In retrieve_context():
result = daemon._search(query, project="longmemeval", rerank=True)
#                                                      ^^^^^ change
```

#### I5.2: Increase Retrieval Pool
Expand top-K from 5 to 20 for initial retrieval, then rerank to top-5:
- More candidates = higher chance of finding answer-bearing turns
- Cross-encoder precision ensures top-5 are the best

---

### Phase I6: Abstention Capability
**Expected lift**: +3-5% (30 abstention questions)
**Effort**: ~100 LOC

#### I6.1: Confidence-Based Abstention
If retrieval confidence is below threshold AND no relevant hits:
```python
if max_similarity < 0.3 and all hits are generic:
    hypothesis = "I don't have information about that in our conversation history."
```

#### I6.2: Abstention Prompt
```
"If the information was NOT discussed in the conversation history,
 say 'This was not discussed in our conversations.'
 Do NOT make up an answer."
```

---

### Phase I7: Reflection & Consolidation (Advanced)
**Expected lift**: +10-15% (what separates 75% from 90%+)
**Effort**: ~500 LOC
**Why last**: Highest complexity, requires architectural changes.

#### I7.1: Session Summarization Pipeline
After ingesting all sessions, run a consolidation pass:
```
All sessions → Group by topic → Generate topic summaries
  "Food preferences: User loves Italian, favorite restaurant Olive Garden,
   tried Thai food in March but didn't enjoy it."
  "Car ownership: Bought Toyota Corolla Jan 2023, first service Feb 2023,
   GPS issue after service, resolved by dealer."
```

#### I7.2: User Profile Construction
Build a structured user profile from all sessions:
```json
{
  "preferences": {"food": "Italian", "restaurant": "Olive Garden"},
  "possessions": {"car": "Toyota Corolla", "phone": "Samsung Galaxy S22"},
  "events": [
    {"date": "2023-01", "event": "Bought car"},
    {"date": "2023-02", "event": "Car first service, GPS issue"}
  ]
}
```

#### I7.3: Multi-Level Retrieval
Like TiMem's temporal memory tree:
1. First check user profile (fast, structured)
2. Then check topic summaries (medium, semantic)
3. Then check individual sessions (detailed, for complex questions)

---

## Implementation Priority Matrix

```
IMPACT vs EFFORT
════════════════════════════════════════════════
                          HIGH IMPACT
                              │
  I5: Cross-encoder    ◄──── │ ────► I1: Session retrieval
  (already built,             │      + index expansion
   just enable)               │      (~300 LOC, +20-30%)
  (~30 LOC, +3-5%)            │
                              │
  I2: Model upgrade    ◄──── │ ────► I4: Knowledge updates
  (~50 LOC, +15-20%)         │      (~200 LOC, +10-15%)
                              │
LOW EFFORT ───────────────────┼──────────────── HIGH EFFORT
                              │
  I6: Abstention       ◄──── │ ────► I7: Reflection
  (~100 LOC, +3-5%)          │      (~500 LOC, +10-15%)
                              │
  I3: Temporal         ◄──── │
  (~150 LOC, +7-11%)         │
                              │
                          LOW IMPACT
```

**Recommended order**: I5 → I2 → I1 → I3 → I4 → I6 → I7

---

## Projected Accuracy by Phase

```
Phase       Cumulative     Questions Correct    Notes
─────────────────────────────────────────────────────────
Baseline    5.0%           25/500               Current
+I5         8-10%          40-50/500            Enable cross-encoder
+I2         20-30%         100-150/500          Upgrade to qwen2.5:7b + CoT
+I1         40-55%         200-275/500          Session retrieval + expansion
+I3         47-60%         235-300/500          Temporal query expansion
+I4         55-70%         275-350/500          Knowledge update detection
+I6         58-73%         290-365/500          Abstention
+I7         68-80%         340-400/500          Reflection + consolidation
```

**With larger model (14b+ or API)**: Each phase gets +5-10% boost.

---

## Quick Wins (Can Do Today)

1. **Enable cross-encoder reranking** — 1 line change, +3-5%
2. **Upgrade generator to qwen2.5:7b** — config change, +10-15%
3. **Add CoT prompting** — prompt change, +5-8%
4. **Increase top-K to 20** — config change, +2-3%

Combined quick wins: **5% → 20-30%** with ~50 LOC changes.

---

## Hardware Considerations (Mac Mini M4 16GB)

| Configuration | RAM Usage | Feasible? |
|--------------|-----------|-----------|
| BGE-M3 + qwen2.5:3b | ~5GB | Yes (current) |
| BGE-M3 + qwen2.5:7b | ~9GB | Yes |
| BGE-M3 + qwen2.5:14b | ~13GB | Tight but feasible |
| BGE-M3 + cross-encoder + qwen2.5:7b | ~10GB | Yes |
| BGE-M3 + cross-encoder + qwen2.5:14b | ~14GB | Tight |
| BGE-M3 + cross-encoder + llama3.1:70b | >40GB | No (need cloud) |

**Sweet spot**: BGE-M3 + cross-encoder + qwen2.5:7b (~10GB) — leaves headroom.

---

## Key Architectural Insight

The top systems (Observational Memory, Hindsight) don't just retrieve better — they **pre-process memories into structured, searchable forms**. The pipeline is:

```
Raw conversations
       │
       ▼
[Ingestion-time processing]     ← This is what we're missing
  - Session summaries
  - Fact extraction
  - Keyphrase expansion
  - Temporal event indexing
  - User profile construction
       │
       ▼
[Multi-level memory store]
  - User profile (structured facts)
  - Topic summaries (consolidated)
  - Session summaries (searchable)
  - Raw turns (detailed context)
       │
       ▼
[Hybrid retrieval]              ← We have this (RRF fusion)
  - Semantic search
  - BM25 keyword
  - Graph traversal            ← We have this (KG)
  - Temporal filtering         ← We have this (decay model)
       │
       ▼
[Reranking]                    ← We have this (cross-encoder)
       │
       ▼
[Context assembly + CoT]       ← We need this
```

**Our system already has 4 of the 6 components.** The two missing pieces are:
1. **Ingestion-time processing** (session summaries, fact extraction, index expansion)
2. **Context assembly with CoT** (structured prompts, temporal ordering)

These are the highest-ROI improvements.

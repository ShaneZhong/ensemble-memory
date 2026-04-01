#!/usr/bin/env python3
"""longmemeval_runner.py — Run LongMemEval benchmark against our ensemble memory system.

End-to-end pipeline:
  1. Load dataset (skip ingestion if DB already exists)
  2. For each question: retrieve context → generate answer → judge
  3. Report per-ability accuracy + latency

Usage:
    python longmemeval_runner.py eval/longmemeval_oracle.json [options]
"""

import json
import logging
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("longmemeval.runner")

# Make hooks + eval importable
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
_HOOKS_DIR = _PROJECT_ROOT / "hooks"
_DAEMON_DIR = _PROJECT_ROOT / "daemon"
for p in [str(_HOOKS_DIR), str(_EVAL_DIR), str(_DAEMON_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from longmemeval_adapter import DEFAULT_DB_PATH, ingest_dataset, batch_embed_memories
from longmemeval_judge import judge_answer

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GENERATOR_MODEL = os.environ.get("LONGMEMEVAL_GEN_MODEL", "qwen2.5:3b")


# ── Ability mapping ──────────────────────────────────────────────────────────

ABILITY_MAP = {
    "single-session-user": "Information Extraction",
    "single-session-assistant": "Information Extraction",
    "single-session-preference": "Information Extraction",
    "multi-session": "Multi-Session Reasoning",
    "knowledge-update": "Knowledge Updates",
    "temporal-reasoning": "Temporal Reasoning",
}


def get_ability(question_type: str) -> str:
    """Map question type to one of the 5 LongMemEval abilities."""
    if "_abs" in question_type:
        return "Abstention"
    return ABILITY_MAP.get(question_type, "Unknown")


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_context(query: str, db_path: str = DEFAULT_DB_PATH, top_k: int = 10) -> list[dict]:
    """Query our ensemble memory search engine for relevant memories.

    Uses the daemon's _search() logic directly (no HTTP needed).
    Disables temporal score filtering (benchmark memories may be years old).
    """
    import db
    db._DB_PATH_OVERRIDE = db_path

    # Import search function from daemon
    import embedding_daemon as daemon

    # Override daemon's memory loading to use our eval DB
    daemon._memories_cache = None  # bust cache
    daemon._cache_loaded_at = 0

    # Disable temporal score floor for benchmark (memories from 2023 would be ~0)
    original_min_temporal = daemon.MIN_TEMPORAL_SCORE
    daemon.MIN_TEMPORAL_SCORE = 0.0

    try:
        result = daemon._search(query, project="longmemeval", rerank=False)
    finally:
        daemon.MIN_TEMPORAL_SCORE = original_min_temporal

    hits = result.get("hits", [])
    return hits[:top_k]


def format_context(hits: list[dict], max_chars: int = 4000) -> str:
    """Format retrieved memories as context for the generator LLM."""
    if not hits:
        return "No relevant conversation history found."

    parts = ["Here is relevant conversation history:\n"]
    char_count = len(parts[0])

    for hit in hits:
        content = hit.get("content", "")
        line = f"- {content}\n"
        if char_count + len(line) > max_chars:
            break
        parts.append(line)
        char_count += len(line)

    return "".join(parts)


# ── Generation ───────────────────────────────────────────────────────────────

def generate_answer(question: str, context: str, question_date: str = "") -> str:
    """Generate an answer to the question using Ollama, given retrieved context."""
    date_hint = f"\nThe current date is {question_date}." if question_date else ""

    prompt = f"""You are a helpful assistant with access to conversation history.
Based on the conversation history provided, answer the user's question.
If the information is not in the history, say "I don't have that information."{date_hint}

{context}

Question: {question}

Answer concisely:"""

    try:
        payload = json.dumps({
            "model": GENERATOR_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        return result.get("response", "").strip()

    except Exception as exc:
        logger.warning("Generation failed: %s", exc)
        return f"ERROR: {exc}"


# ── Main runner ──────────────────────────────────────────────────────────────

def run_eval(
    dataset_path: str,
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
    skip_ingest: bool = False,
    output_path: str | None = None,
) -> dict:
    """Run the full LongMemEval evaluation pipeline.

    Returns:
        {
            "overall_accuracy": float,
            "per_ability": {ability: {"correct": int, "total": int, "accuracy": float}},
            "avg_retrieval_ms": float,
            "avg_generation_ms": float,
            "results": [per-question details]
        }
    """
    # Step 1: Ingest dataset
    if not skip_ingest:
        logger.info("Ingesting dataset...")
        stats = ingest_dataset(dataset_path, db_path, limit)
        logger.info("Ingestion: %s", stats)

        logger.info("Batch embedding memories...")
        embedded = batch_embed_memories(db_path)
        logger.info("Embedded %d memories", embedded)
    else:
        logger.info("Skipping ingestion (--skip-ingest)")

    # Step 2: Load questions (stratified sample if limited)
    with open(dataset_path) as f:
        dataset = json.load(f)
    if limit and limit < len(dataset):
        # Stratified sample: pick proportionally from each question type
        import random
        random.seed(42)
        by_type = defaultdict(list)
        for q in dataset:
            by_type[q["question_type"]].append(q)
        sampled = []
        total_types = len(by_type)
        per_type = max(1, limit // total_types)
        for qtype, questions in sorted(by_type.items()):
            n = min(per_type, len(questions))
            sampled.extend(random.sample(questions, n))
        # Fill remaining slots
        remaining = limit - len(sampled)
        if remaining > 0:
            used_ids = {q["question_id"] for q in sampled}
            pool = [q for q in dataset if q["question_id"] not in used_ids]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))
        dataset = sampled[:limit]
        random.shuffle(dataset)

    logger.info("Running %d questions...", len(dataset))

    # Step 3: Process each question
    results = []
    per_ability = defaultdict(lambda: {"correct": 0, "total": 0})
    total_retrieval_ms = 0
    total_generation_ms = 0
    total_judge_ms = 0

    for i, q in enumerate(dataset):
        question_id = q["question_id"]
        question_type = q["question_type"]
        question_text = q["question"]
        ground_truth = q["answer"]
        question_date = q.get("question_date", "")
        ability = get_ability(question_type)

        # Retrieve
        t0 = time.time()
        hits = retrieve_context(question_text, db_path)
        retrieval_ms = (time.time() - t0) * 1000
        total_retrieval_ms += retrieval_ms

        context = format_context(hits)

        # Generate
        t0 = time.time()
        hypothesis = generate_answer(question_text, context, question_date)
        generation_ms = (time.time() - t0) * 1000
        total_generation_ms += generation_ms

        # Judge
        t0 = time.time()
        judge_result = judge_answer(question_text, ground_truth, hypothesis, question_type)
        judge_ms = (time.time() - t0) * 1000
        total_judge_ms += judge_ms

        label = judge_result["label"]
        per_ability[ability]["total"] += 1
        per_ability[ability]["correct"] += label

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "ability": ability,
            "question": question_text,
            "answer": ground_truth,
            "hypothesis": hypothesis,
            "judge_label": label,
            "judge_raw": judge_result["raw_response"],
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "judge_ms": round(judge_ms, 1),
            "num_hits": len(hits),
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            correct_so_far = sum(r["judge_label"] for r in results)
            logger.info(
                "[%d/%d] accuracy=%.1f%% retrieval=%.0fms gen=%.0fms",
                i + 1, len(dataset),
                100.0 * correct_so_far / len(results),
                retrieval_ms, generation_ms,
            )

    # Step 4: Compute metrics
    total_correct = sum(r["judge_label"] for r in results)
    overall_accuracy = total_correct / len(results) if results else 0.0

    ability_report = {}
    for ability, counts in sorted(per_ability.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        ability_report[ability] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": round(acc, 4),
        }

    report = {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_correct": total_correct,
        "total_questions": len(results),
        "per_ability": ability_report,
        "avg_retrieval_ms": round(total_retrieval_ms / len(results), 1) if results else 0,
        "avg_generation_ms": round(total_generation_ms / len(results), 1) if results else 0,
        "avg_judge_ms": round(total_judge_ms / len(results), 1) if results else 0,
        "results": results,
    }

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)

    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 60)
    print("LongMemEval Evaluation Report")
    print("=" * 60)
    print(f"\nOverall Accuracy: {report['overall_accuracy']:.1%} "
          f"({report['total_correct']}/{report['total_questions']})")
    print(f"\nAvg Retrieval Latency: {report['avg_retrieval_ms']:.0f}ms")
    print(f"Avg Generation Latency: {report['avg_generation_ms']:.0f}ms")
    print(f"Avg Judge Latency: {report['avg_judge_ms']:.0f}ms")
    print(f"\n{'Ability':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 53)
    for ability, data in sorted(report["per_ability"].items()):
        print(f"{ability:<25} {data['correct']:>8} {data['total']:>6} {data['accuracy']:>9.1%}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("dataset_path", help="Path to longmemeval_oracle.json")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not args.output:
        args.output = str(
            _EVAL_DIR / f"longmemeval_results_{int(time.time())}.json"
        )

    report = run_eval(
        args.dataset_path,
        db_path=args.db_path,
        limit=args.limit,
        skip_ingest=args.skip_ingest,
        output_path=args.output,
    )
    print_report(report)

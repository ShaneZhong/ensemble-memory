#!/usr/bin/env python3
"""analyze_retrieval.py — Analyze retrieval quality from LongMemEval results.

Measures Recall@K: does the retrieved context contain information needed to
answer the question? This isolates retrieval quality from LLM reasoning ability.

Usage:
    python analyze_retrieval.py <results_json>
"""

import json
import sys
from collections import defaultdict


def keyword_overlap(answer: str, context: str) -> float:
    """Compute word overlap between ground-truth answer and retrieved context.

    Returns fraction of answer words found in context.
    """
    answer_words = set(str(answer).lower().split())
    context_words = set(str(context).lower().split())

    # Remove common stopwords
    stopwords = {
        "the", "a", "an", "is", "in", "it", "to", "and", "or", "for",
        "of", "on", "at", "be", "was", "has", "i", "my", "we", "you",
        "that", "this", "with", "from", "also", "not", "are", "were",
        "been", "have", "had", "do", "does", "did", "will", "would",
        "could", "should", "can", "may", "might", "but", "so", "if",
    }
    answer_words -= stopwords
    context_words -= stopwords

    if not answer_words:
        return 0.0

    return len(answer_words & context_words) / len(answer_words)


def analyze_results(results_path: str) -> None:
    """Analyze retrieval quality from evaluation results."""
    with open(results_path) as f:
        report = json.load(f)

    results = report.get("results", [])
    if not results:
        print("No results found.")
        return

    # Build per-question retrieval analysis
    per_ability = defaultdict(lambda: {
        "total": 0,
        "has_hits": 0,
        "answer_in_context": 0,
        "avg_overlap": 0.0,
        "overlaps": [],
        "correct_by_judge": 0,
    })

    no_hits_questions = []
    low_overlap_questions = []

    for r in results:
        ability = r.get("ability", "Unknown")
        stats = per_ability[ability]
        stats["total"] += 1
        stats["correct_by_judge"] += r.get("judge_label", 0)

        num_hits = r.get("num_hits", 0)
        if num_hits > 0:
            stats["has_hits"] += 1
        else:
            no_hits_questions.append(r)

        # Check if answer keywords appear in retrieved context
        answer = r.get("answer", "")
        context = r.get("retrieved_context", "")
        if not context:
            # Fallback to hypothesis if context not saved
            context = r.get("hypothesis", "")

        overlap = keyword_overlap(answer, context)
        stats["overlaps"].append(overlap)

        # Consider "answer found" if >50% of answer words appear
        if overlap >= 0.5:
            stats["answer_in_context"] += 1
        elif overlap < 0.3 and num_hits > 0:
            low_overlap_questions.append(r)

    # Print report
    print("\n" + "=" * 70)
    print("LongMemEval — Retrieval Quality Analysis")
    print("=" * 70)

    print(f"\nTotal questions: {len(results)}")
    print(f"Questions with hits: {sum(s['has_hits'] for s in per_ability.values())}/{len(results)}")
    print(f"Questions with no hits: {len(no_hits_questions)}")

    print(f"\n{'Ability':<25} {'Total':>6} {'Hits':>6} {'Ans Found':>10} {'Avg Overlap':>12} {'Judge Acc':>10}")
    print("-" * 75)

    for ability in sorted(per_ability.keys()):
        stats = per_ability[ability]
        avg_overlap = (
            sum(stats["overlaps"]) / len(stats["overlaps"])
            if stats["overlaps"]
            else 0.0
        )
        hit_rate = stats["has_hits"] / stats["total"] if stats["total"] > 0 else 0
        ans_rate = stats["answer_in_context"] / stats["total"] if stats["total"] > 0 else 0
        judge_acc = stats["correct_by_judge"] / stats["total"] if stats["total"] > 0 else 0

        print(
            f"{ability:<25} {stats['total']:>6} "
            f"{stats['has_hits']:>6} "
            f"{stats['answer_in_context']:>10} "
            f"{avg_overlap:>11.1%} "
            f"{judge_acc:>9.1%}"
        )

    # Diagnosis
    print("\n" + "-" * 70)
    print("DIAGNOSIS")
    print("-" * 70)

    total_with_hits = sum(s["has_hits"] for s in per_ability.values())
    total_ans_found = sum(s["answer_in_context"] for s in per_ability.values())
    total_correct = sum(s["correct_by_judge"] for s in per_ability.values())

    if total_with_hits < len(results) * 0.8:
        print(f"  RETRIEVAL GAP: Only {total_with_hits}/{len(results)} questions got hits.")
        print("  → Check embedding quality, similarity thresholds, or search coverage.")
    else:
        print(f"  RETRIEVAL: {total_with_hits}/{len(results)} questions got hits (good).")

    if total_ans_found < total_with_hits * 0.5:
        print(f"  RELEVANCE GAP: Only {total_ans_found}/{total_with_hits} hits contained answer keywords.")
        print("  → Retrieval is finding memories but not the RIGHT ones.")
        print("  → Consider: more context (top-K), better ranking, or cross-encoder reranking.")
    else:
        print(f"  RELEVANCE: {total_ans_found}/{total_with_hits} hits contained answer keywords.")

    if total_correct < total_ans_found * 0.5:
        print(f"  GENERATION GAP: {total_correct} correct vs {total_ans_found} with answer available.")
        print("  → LLM (qwen2.5:3b) can't synthesize answers from context.")
        print("  → Upgrade to larger model for accurate scoring.")
    elif total_ans_found > 0:
        print(f"  GENERATION: {total_correct}/{total_ans_found} correct when answer available.")

    # Sample failures
    if low_overlap_questions:
        print(f"\n  Sample questions where retrieval missed (low overlap, {len(low_overlap_questions)} total):")
        for q in low_overlap_questions[:3]:
            print(f"    Q: {str(q['question'])[:80]}")
            print(f"    A: {str(q['answer'])[:80]}")
            print(f"    H: {str(q['hypothesis'])[:80]}")
            print()

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_retrieval.py <results.json>")
        sys.exit(1)
    analyze_results(sys.argv[1])

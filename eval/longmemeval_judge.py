#!/usr/bin/env python3
"""longmemeval_judge.py — LLM-as-judge scoring for LongMemEval answers.

Adapted from LongMemEval's evaluate_qa.py prompts, using Ollama instead of GPT-4.
"""

import json
import logging
import urllib.request
import os

logger = logging.getLogger("longmemeval.judge")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("LONGMEMEVAL_JUDGE_MODEL", "qwen2.5:3b")


# ── Judge prompts (adapted from LongMemEval) ────────────────────────────────

JUDGE_PROMPT_DEFAULT = """I will give you a question, a correct answer, and a response from a model.
Please answer "yes" if the response contains the correct answer, and "no" otherwise.
Do not explain. Just answer "yes" or "no".

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Does the model's response contain the correct answer?"""

JUDGE_PROMPT_TEMPORAL = """I will give you a question about timing/dates, a correct answer, and a response from a model.
Please answer "yes" if the response is correct (allow off-by-one errors for dates and minor paraphrasing), and "no" otherwise.
Do not explain. Just answer "yes" or "no".

Question: {question}
Correct Answer: {answer}
Model Response: {hypothesis}

Does the model's response contain the correct answer?"""

JUDGE_PROMPT_KNOWLEDGE_UPDATE = """I will give you a question, the LATEST correct answer (information may have been updated), and a response from a model.
The correct answer reflects the most recent information. The model should give the latest answer, not an older one.
Please answer "yes" if the response contains the latest correct answer, and "no" otherwise.
Do not explain. Just answer "yes" or "no".

Question: {question}
Latest Correct Answer: {answer}
Model Response: {hypothesis}

Does the model's response contain the latest correct answer?"""

JUDGE_PROMPT_ABSTENTION = """I will give you a question and a response from a model.
The question asks about something that was NEVER mentioned in the conversation history.
The model should refuse to answer, say it doesn't know, or indicate the information wasn't discussed.
Please answer "yes" if the model correctly abstains/refuses, and "no" if the model makes up an answer.
Do not explain. Just answer "yes" or "no".

Question: {question}
Model Response: {hypothesis}

Does the model correctly abstain or refuse to answer?"""


def _get_judge_prompt(question_type: str) -> str:
    """Select the appropriate judge prompt based on question type."""
    if "temporal" in question_type:
        return JUDGE_PROMPT_TEMPORAL
    if "knowledge-update" in question_type:
        return JUDGE_PROMPT_KNOWLEDGE_UPDATE
    if "_abs" in question_type:
        return JUDGE_PROMPT_ABSTENTION
    return JUDGE_PROMPT_DEFAULT


def judge_answer(
    question: str,
    answer: str,
    hypothesis: str,
    question_type: str,
) -> dict:
    """Score a model's answer against ground truth using Ollama as judge.

    Returns:
        {"label": 1|0, "raw_response": str, "judge_model": str}
    """
    prompt_template = _get_judge_prompt(question_type)

    if "_abs" in question_type:
        prompt = prompt_template.format(question=question, hypothesis=hypothesis)
    else:
        prompt = prompt_template.format(
            question=question, answer=answer, hypothesis=hypothesis
        )

    try:
        payload = json.dumps({
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        raw_response = result.get("response", "").strip().lower()

        # Parse yes/no
        if raw_response.startswith("yes"):
            label = 1
        elif raw_response.startswith("no"):
            label = 0
        else:
            # Fuzzy: check if "yes" appears more prominently
            label = 1 if "yes" in raw_response and "no" not in raw_response else 0

        return {
            "label": label,
            "raw_response": raw_response,
            "judge_model": JUDGE_MODEL,
        }

    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return {
            "label": 0,
            "raw_response": f"ERROR: {exc}",
            "judge_model": JUDGE_MODEL,
        }


def judge_batch(results: list[dict]) -> list[dict]:
    """Score a batch of results. Each entry must have:
    question, answer, hypothesis, question_type, question_id.

    Returns the same list with 'judge' field added.
    """
    for entry in results:
        entry["judge"] = judge_answer(
            question=entry["question"],
            answer=entry.get("answer", ""),
            hypothesis=entry.get("hypothesis", ""),
            question_type=entry["question_type"],
        )
    return results

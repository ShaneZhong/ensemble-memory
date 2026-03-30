#!/usr/bin/env python3
"""Regex triage — detect correction/decision signals in a conversation turn.

Usage: triage.py <temp_file_path>
Output: JSON array of signals to stdout, or [] if none found.
Latency: < 5ms (stdlib only, no external deps).
"""
import json
import re
import sys

TIER1_CORRECTION = [
    r"\bno[,.]?\s+(not|don't|never|stop|wrong)",
    r"\bactually[,.]?\s",
    r"\bdon'?t\s+(use|do|add|make|create)",
    r"\bstop\s+(doing|using|adding)",
    r"\bthat'?s\s+(wrong|incorrect|not right)",
    r"\binstead[,.]?\s+(use|try|do)",
    r"\bnever\s+(use|do|add|make|create|import|call|run)\s+\S+.{0,30}(for|when|in)\b",
    r"(?:^|\n|[.!]\s+)always\s+(use|do|add|run|prefer|set|ensure)\b",
    r"\buse\s+\S+[,.]?\s+not\s+\S+",
    r"\bnot\s+\S+[,.]\s+use\s+\S+",
]

TIER4_DECISION = [
    r"\blet'?s\s+(use|go with|switch to|try)",
    r"\bwe('ll| will| should)\s+(use|go with)",
    r"\bdecided\s+(to|on|that)",
    r"\bthe plan is\s+to",
    r"\bfrom now on[,.]?\s+(use|do|always)",
    r"\bgoing forward[,.]?\s+(use|do|always|we)",
    r"\bremember[,.]?\s+(always|never|use|don't)",
]

SIGNALS = [
    (1, "correction", TIER1_CORRECTION),
    (4, "decision", TIER4_DECISION),
]

# Compiled patterns (done once at module load)
COMPILED = [
    (tier, sig_type, pattern, re.compile(pattern, re.IGNORECASE))
    for tier, sig_type, patterns in SIGNALS
    for pattern in patterns
]


def extract_user_text(text: str) -> str:
    """Return only the user-authored portions of a conversation turn.

    Handles two common formats:
    1. Role-prefixed lines: lines starting with 'Human:' or 'User:'
    2. Plain text (no role markers): treat the whole thing as user text.
    """
    lines = text.splitlines()
    user_lines = []
    in_user_block = False
    has_role_markers = any(
        ln.startswith(("Human:", "User:", "Assistant:", "Claude:"))
        for ln in lines
    )

    if not has_role_markers:
        return text

    for line in lines:
        if line.startswith(("Human:", "User:")):
            in_user_block = True
            user_lines.append(line)
        elif line.startswith(("Assistant:", "Claude:")):
            in_user_block = False
        elif in_user_block:
            user_lines.append(line)

    return "\n".join(user_lines)


def triage(text: str) -> list[dict]:
    user_text = extract_user_text(text)
    signals = []
    for tier, sig_type, pattern, compiled in COMPILED:
        m = compiled.search(user_text)
        if m:
            signals.append({
                "tier": tier,
                "type": sig_type,
                "match": m.group(0),
            })
    return signals


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: triage.py <temp_file_path>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read()

    signals = triage(text)
    print(json.dumps(signals))


if __name__ == "__main__":
    main()

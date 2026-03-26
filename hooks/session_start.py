#!/usr/bin/env python3
"""session_start.py — Load standing context from ensemble memory DB at session start.

Usage: session_start.py <session_id>

Env vars:
    ENSEMBLE_MEMORY_PROJECT  Override project path (default: cwd)

Outputs JSON to stdout:
    {"additionalContext": "## Ensemble Memory: Standing Context\n..."}
    or {} if no memories found or DB does not exist yet.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Locate hooks dir and make db importable ───────────────────────────────────
HOOKS_DIR = Path(__file__).resolve().parent
if str(HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(HOOKS_DIR))

# ── Memory type display config ────────────────────────────────────────────────
TYPE_HEADERS = {
    "correction": "### Corrections (do NOT repeat these mistakes)",
    "procedural": "### Procedural Rules",
    "semantic":   "### Project Facts",
    "episodic":   "### Recent Episodes",
}

TYPE_ORDER = ["correction", "procedural", "semantic", "episodic"]

MIN_IMPORTANCE = 7  # Configurable: load memories at or above this importance score


def format_context(memories: list[dict], project: str) -> str:
    """Format a list of memory dicts into the standing context string."""
    # Group by memory_type
    grouped: dict[str, list[dict]] = {t: [] for t in TYPE_ORDER}
    for mem in memories:
        mtype = mem.get("memory_type", "semantic")
        if mtype not in grouped:
            mtype = "semantic"
        grouped[mtype].append(mem)

    lines = [
        "## Ensemble Memory: Standing Rules & Corrections",
        "",
        "IMPORTANT: The following are learned corrections and rules from previous sessions.",
        "You MUST follow these. They represent explicit user preferences and past mistakes",
        "that were corrected. Violating these will frustrate the user.",
        "",
    ]

    any_section = False
    for mtype in TYPE_ORDER:
        mems = grouped[mtype]
        if not mems:
            continue
        any_section = True
        lines.append(TYPE_HEADERS[mtype])
        for mem in mems:
            importance = mem.get("importance", 0)
            content = mem.get("content", "").strip()
            lines.append(f"- **[MUST FOLLOW]** {content}")
        lines.append("")

    if not any_section:
        return ""

    # Footer
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(
        f"Last updated: {now_str} | "
        f"Memories loaded: {len(memories)} | "
        f"Project: {project}"
    )

    return "\n".join(lines)


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else ""
    project = os.environ.get("ENSEMBLE_MEMORY_PROJECT", "") or os.getcwd()

    # Try to import db — if it doesn't exist yet, exit gracefully with {}
    try:
        import db
    except ImportError:
        print("{}")
        return

    # Load memories from DB — gracefully handle missing DB or any DB error
    try:
        memories = db.get_memories_for_session_start(
            project=project,
            min_importance=MIN_IMPORTANCE,
        )
    except Exception:
        print("{}")
        return

    if not memories:
        # Record the session start even if no memories were loaded
        if session_id:
            try:
                db.record_session(session_id=session_id, project=project)
            except Exception:
                pass
        print("{}")
        return

    # Record session start in DB
    if session_id:
        try:
            db.record_session(session_id=session_id, project=project)
        except Exception:
            pass

    context = format_context(memories, project)
    if not context:
        print("{}")
        return

    result = {"additionalContext": context}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

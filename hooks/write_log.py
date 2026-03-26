#!/usr/bin/env python3
"""write_log.py — Append extraction results to a daily markdown memory log.

Usage: write_log.py <extraction_json> <session_id>

Env vars:
    ENSEMBLE_MEMORY_LOGS  Daily .md output dir  (default: ~/.ensemble_memory/memory/)
    ENSEMBLE_MEMORY_DIR   User data root        (default: ~/.ensemble_memory/)
    TRANSCRIPT_PATH       Source transcript path (metadata only)
"""

import fcntl, json, os, sys
from datetime import datetime
from pathlib import Path


def _dir(env: str, rel: str) -> Path:
    v = os.environ.get(env, "")
    p = Path(v).expanduser() if v else Path("~/.ensemble_memory").expanduser() / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ts(existing: str, base: str) -> str:
    """Return HH:MM, appending :01/:02 suffix on collision."""
    ts, n = base, 0
    while f"### {ts}" in existing:
        n += 1; ts = f"{base}:{n:02d}"
    return ts


def _bullets(mem: dict) -> list[str]:
    t, content, rule = mem.get("type",""), mem.get("content","").strip(), mem.get("rule")
    wrong, resolved = mem.get("what_went_wrong"), mem.get("how_it_was_resolved")
    lines = []
    if t == "correction":
        lines.append(f"- User corrected: {wrong or content}")
        if resolved: lines.append(f"- Resolution: {resolved}")
        if rule:     lines.append(f"- Rule derived: {rule}")
    elif t == "procedural":
        lines.append(f"- Procedure: {content}")
        if rule: lines.append(f"- Rule: {rule}")
    else:
        lines.append(f"- {content}")
    return lines


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: write_log.py <extraction_json> <session_id>")

    extraction_raw, session_id = sys.argv[1], sys.argv[2]
    transcript_path = os.environ.get("TRANSCRIPT_PATH", "")

    try:
        extraction = json.loads(extraction_raw)
    except json.JSONDecodeError as e:
        sys.exit(f"write_log: invalid JSON: {e}")

    memories = extraction.get("memories", [])
    if not memories:
        return

    # Dedup: read existing log and skip memories whose content is already present
    logs_dir_check = _dir("ENSEMBLE_MEMORY_LOGS", "memory")
    date_check = datetime.now().strftime("%Y-%m-%d")
    existing_log_path = logs_dir_check / f"{date_check}.md"
    existing_content = ""
    if existing_log_path.exists():
        existing_content = existing_log_path.read_text(encoding="utf-8")

    new_memories = []
    for mem in memories:
        content = mem.get("content", "").strip()
        if not content:
            continue
        # Skip if this content (or a close substring) is already in today's log
        if content in existing_content:
            continue
        # Also check if first 50 chars match (catches minor rewording)
        if len(content) > 50 and content[:50] in existing_content:
            continue
        new_memories.append(mem)

    if not new_memories:
        return
    memories = new_memories

    logs_dir = _dir("ENSEMBLE_MEMORY_LOGS", "memory")
    data_dir = _dir("ENSEMBLE_MEMORY_DIR", "")

    now = datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M")

    # Raw extraction JSONL
    ex_dir = data_dir / "logs" / "extractions"
    ex_dir.mkdir(parents=True, exist_ok=True)
    with open(ex_dir / f"{date_str}.jsonl", "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps({"session_id": session_id, "timestamp": now.isoformat(), **extraction}, ensure_ascii=False) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)

    # Daily markdown log
    md_path = logs_dir / f"{date_str}.md"
    with open(md_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0); existing = fh.read()
            out = []
            if not existing.strip():
                out.append(f"# {date_str}\n")
            session_heading = f"## Session: {session_id}"
            if session_heading not in existing:
                out.append(f"\n{session_heading}\n")
            out.append(f"\n### {_ts(existing + ''.join(out), time_str)}")
            # HTML comment: machine-readable metadata (invisible in rendered markdown)
            m0 = memories[0]
            meta = f"session:{session_id}"
            if "type"       in m0: meta += f" type:{m0['type']}"
            if "importance" in m0: meta += f" importance:{m0['importance']}"
            if "confidence" in m0: meta += f"\n     confidence:{m0['confidence']}"
            if transcript_path:    meta += f" transcript:{transcript_path}"
            out.append(f"<!-- {meta} -->")
            for mem in memories:
                out.extend(_bullets(mem))
            out.append("")
            fh.seek(0, 2)
            fh.write("\n".join(out) + "\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "agent_memory" / "term_highlights.json"
ALLOWED_CATEGORIES = {"math", "method", "result", "concept"}
_MEMORY_LOCK = threading.RLock()


def term_memory_path() -> Path:
    configured = os.getenv("PAPER_TERM_MEMORY_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MEMORY_PATH


def term_memory_prompt_path(path: Path | None = None) -> Path:
    return (path or term_memory_path()).with_suffix(".md")


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip(" \t\r\n.,;:()[]{}")


def _empty_memory() -> dict[str, Any]:
    return {"version": 1, "updated_at": None, "terms": {}}


def load_term_memory(path: Path | None = None) -> dict[str, Any]:
    target = path or term_memory_path()
    if not target.exists():
        return _empty_memory()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_memory()
    if not isinstance(payload, dict) or not isinstance(payload.get("terms"), dict):
        return _empty_memory()
    return payload


def load_learned_terms(path: Path | None = None) -> dict[str, dict[str, Any]]:
    payload = load_term_memory(path)
    return {
        str(key).lower(): value
        for key, value in payload.get("terms", {}).items()
        if isinstance(value, dict) and value.get("term")
    }


def _write_prompt_memory(payload: dict[str, Any], path: Path) -> None:
    records = [record for record in payload.get("terms", {}).values() if isinstance(record, dict)]
    records.sort(
        key=lambda item: (int(item.get("correction_count", 0)), str(item.get("last_seen_at", ""))),
        reverse=True,
    )
    lines = [
        "# User-Confirmed Paper Terms",
        "",
        "Treat these human-confirmed terms as high-confidence highlighting signals.",
        "Keep paper evidence, web context, and inference separate when explaining them.",
        "",
    ]
    if records:
        lines.extend(
            f"- {record.get('term')} [{record.get('category', 'concept')}], corrections: {record.get('correction_count', 1)}"
            for record in records[:80]
        )
    else:
        lines.append("No user-confirmed terms have been recorded yet.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_term_memory_files(path: Path | None = None) -> tuple[Path, Path]:
    target = path or term_memory_path()
    with _MEMORY_LOCK:
        payload = load_term_memory(target)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        prompt_path = term_memory_prompt_path(target)
        _write_prompt_memory(payload, prompt_path)
    return target, prompt_path


def remember_term(
    term: str,
    category: str,
    *,
    paper_title: str | None = None,
    source_pdf: str | None = None,
    page: int | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    clean_term = normalize_term(term)
    clean_category = category.strip().lower()
    if len(clean_term) < 2 or len(clean_term) > 120:
        raise ValueError("A remembered term must be between 2 and 120 characters.")
    if clean_category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unknown highlight category: {category}")
    if any(ord(char) < 32 for char in clean_term):
        raise ValueError("A remembered term cannot contain control characters.")

    target = path or term_memory_path()
    now = datetime.now(timezone.utc).isoformat()
    key = clean_term.lower()
    with _MEMORY_LOCK:
        payload = load_term_memory(target)
        terms = payload.setdefault("terms", {})
        record = terms.get(key) if isinstance(terms.get(key), dict) else {}
        examples = record.get("examples", []) if isinstance(record.get("examples"), list) else []
        example = {
            "paper_title": paper_title,
            "source_pdf": source_pdf,
            "page": page,
            "recorded_at": now,
        }
        signature = (paper_title or "", source_pdf or "", page)
        existing_signatures = {
            (item.get("paper_title") or "", item.get("source_pdf") or "", item.get("page"))
            for item in examples
            if isinstance(item, dict)
        }
        if signature not in existing_signatures:
            examples.append(example)
        record = {
            "term": clean_term,
            "category": clean_category,
            "correction_count": int(record.get("correction_count", 0)) + 1,
            "first_seen_at": record.get("first_seen_at") or now,
            "last_seen_at": now,
            "examples": examples[-20:],
        }
        terms[key] = record
        payload["updated_at"] = now
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, target)
        _write_prompt_memory(payload, term_memory_prompt_path(target))
        return record

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_agent.workspace import Workspace


FEEDBACK_VERSION = 1
FEEDBACK_RELATIVE_PATH = "memory/term_relevance_feedback.json"
ALLOWED_LABELS = {"relevant", "irrelevant"}
_FEEDBACK_LOCK = threading.RLock()


def relevance_key(term: str) -> str:
    """Normalize harmless punctuation variants without merging different concepts."""

    return "".join(re.findall(r"[a-z0-9]+", str(term).casefold()))


def feedback_path(workspace: Workspace) -> Path:
    return workspace.path(FEEDBACK_RELATIVE_PATH)


def empty_feedback() -> dict[str, Any]:
    return {"version": FEEDBACK_VERSION, "updated_at": None, "labels": {}}


def load_term_feedback(workspace: Workspace) -> dict[str, Any]:
    path = feedback_path(workspace)
    if not path.exists():
        return empty_feedback()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_feedback()
    if (
        not isinstance(payload, dict)
        or payload.get("version") != FEEDBACK_VERSION
        or not isinstance(payload.get("labels"), dict)
    ):
        return empty_feedback()
    return payload


def record_term_feedback(
    workspace: Workspace,
    term: str,
    label: str,
    *,
    category: str | None = None,
    score: float | None = None,
    vocabulary_source: str | None = None,
    page: int | None = None,
) -> dict[str, Any]:
    clean_term = re.sub(r"\s+", " ", str(term)).strip(" \t\r\n.,;:()[]{}")
    clean_label = str(label).strip().casefold()
    key = relevance_key(clean_term)
    if not 2 <= len(clean_term) <= 120 or not key:
        raise ValueError("A feedback term must be between 2 and 120 readable characters.")
    if clean_label not in ALLOWED_LABELS:
        raise ValueError(f"Unknown term relevance label: {label}")

    path = feedback_path(workspace)
    with _FEEDBACK_LOCK:
        payload = load_term_feedback(workspace)
        labels = payload.setdefault("labels", {})
        previous = labels.get(key) if isinstance(labels.get(key), dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "term": clean_term,
            "label": clean_label,
            "category": category,
            "score": round(float(score), 3) if score is not None else None,
            "vocabulary_source": vocabulary_source,
            "page": page,
            "first_labeled_at": previous.get("first_labeled_at") or now,
            "last_labeled_at": now,
            "revision_count": int(previous.get("revision_count", 0)) + 1,
        }
        labels[key] = record
        payload["updated_at"] = now
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        return record


def suppressed_relevance_keys(workspace: Workspace) -> set[str]:
    return {
        str(key)
        for key, record in load_term_feedback(workspace).get("labels", {}).items()
        if isinstance(record, dict) and record.get("label") == "irrelevant"
    }

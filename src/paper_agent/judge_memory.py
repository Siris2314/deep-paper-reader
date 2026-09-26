from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from paper_agent.workspace import Workspace


JudgeRating = Literal["agree", "disagree"]
JUDGE_DIMENSIONS = {
    "overall",
    "correctness",
    "paper_grounding",
    "symbol_coverage",
    "latex_fidelity",
    "usefulness",
}
_LOCK = threading.RLock()


def _memory_path() -> Path:
    return Path(__file__).resolve().parents[2] / "agent_memory" / "math_judge_alignment.json"


def _empty_memory() -> dict[str, object]:
    return {"version": 1, "semantic": [], "episodes": []}


def _load_memory() -> dict[str, Any]:
    path = _memory_path()
    if not path.exists():
        return _empty_memory()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_memory()
    if not isinstance(payload, dict):
        return _empty_memory()
    payload.setdefault("version", 1)
    payload.setdefault("semantic", [])
    payload.setdefault("episodes", [])
    return payload


def _tokens(value: object) -> set[str]:
    text = str(value or "").casefold()
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathcal|mathbb|operatorname)", " ", text)
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}", text)
        if token not in {"the", "this", "that", "with", "from", "into", "paper", "equation"}
    }


def _equation_key(explanation: dict[str, Any]) -> str:
    value = "\n".join(
        [
            str(explanation.get("display_latex") or ""),
            str(explanation.get("role") or ""),
            str(explanation.get("page") or ""),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def retrieve_math_judge_memory(
    explanation: dict[str, Any],
    *,
    max_principles: int = 8,
    max_episodes: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    memory = _load_memory()
    query = " ".join(
        [
            str(explanation.get("display_latex") or ""),
            str(explanation.get("role") or ""),
            str(explanation.get("plain_english") or ""),
            str(explanation.get("context_fit") or ""),
            json.dumps(explanation.get("symbols") or [], ensure_ascii=False),
        ]
    )
    query_tokens = _tokens(query)

    principles: list[tuple[float, dict[str, Any]]] = []
    for item in memory.get("semantic", []):
        if not isinstance(item, dict) or not item.get("active", True):
            continue
        overlap = len(query_tokens & _tokens(item.get("principle")))
        uses = max(1, int(item.get("feedbackCount") or 1))
        score = overlap * 2.0 + min(uses, 5) * 0.1
        principles.append((score, item))
    principles.sort(key=lambda pair: (pair[0], str(pair[1].get("updatedAt") or "")), reverse=True)

    episodes: list[tuple[float, dict[str, Any]]] = []
    for item in memory.get("episodes", []):
        if not isinstance(item, dict) or not item.get("active", True):
            continue
        episode_text = " ".join(
            [
                str(item.get("displayLatex") or ""),
                str(item.get("role") or ""),
                str(item.get("feedback") or ""),
            ]
        )
        overlap = len(query_tokens & _tokens(episode_text))
        exact = item.get("equationKey") == _equation_key(explanation)
        if not exact and overlap == 0:
            continue
        episodes.append((100.0 if exact else float(overlap), item))
    episodes.sort(key=lambda pair: (pair[0], str(pair[1].get("createdAt") or "")), reverse=True)

    return {
        "principles": [item for _, item in principles[: max(0, max_principles)]],
        "episodes": [item for _, item in episodes[: max(0, max_episodes)]],
    }


def record_math_judge_feedback(
    workspace: Workspace,
    *,
    explanation: dict[str, Any],
    rating: JudgeRating,
    dimension: str = "overall",
    feedback: str = "",
    remember: bool = True,
) -> dict[str, Any]:
    if rating not in {"agree", "disagree"}:
        raise ValueError("Unsupported math judge rating.")
    clean_dimension = dimension if dimension in JUDGE_DIMENSIONS else "overall"
    clean_feedback = re.sub(r"\s+", " ", feedback).strip()[:4000]
    now = datetime.now(timezone.utc).isoformat()
    evaluation = (
        explanation.get("evaluation") if isinstance(explanation.get("evaluation"), dict) else {}
    )
    record = {
        "id": uuid.uuid4().hex,
        "equationKey": _equation_key(explanation),
        "equationId": str(explanation.get("equation_id") or "")[:200],
        "page": explanation.get("page"),
        "displayLatex": str(explanation.get("display_latex") or "")[:2000],
        "role": str(explanation.get("role") or "")[:500],
        "rating": rating,
        "dimension": clean_dimension,
        "feedback": clean_feedback,
        "judgeVerdict": str(evaluation.get("verdict") or ""),
        "judgeScore": evaluation.get("overall_score"),
        "judgeSummary": str(evaluation.get("summary") or "")[:1200],
        "remembered": bool(remember),
        "active": True,
        "createdAt": now,
    }
    local_path = workspace.path("math/judge_feedback.jsonl")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not remember:
        return record

    with _LOCK:
        memory = _load_memory()
        episodes = [
            item
            for item in memory.get("episodes", [])
            if isinstance(item, dict) and item.get("equationKey") != record["equationKey"]
        ]
        episodes.append(record)
        memory["episodes"] = episodes[-300:]

        if rating == "disagree" and clean_feedback:
            semantic = [item for item in memory.get("semantic", []) if isinstance(item, dict)]
            normalized = clean_feedback.casefold()
            existing = next(
                (
                    item
                    for item in semantic
                    if item.get("dimension") == clean_dimension
                    and str(item.get("principle") or "").casefold() == normalized
                ),
                None,
            )
            if existing:
                existing["feedbackCount"] = int(existing.get("feedbackCount") or 1) + 1
                existing["updatedAt"] = now
                existing["active"] = True
            else:
                semantic.append(
                    {
                        "id": uuid.uuid4().hex,
                        "dimension": clean_dimension,
                        "principle": clean_feedback,
                        "feedbackCount": 1,
                        "active": True,
                        "createdAt": now,
                        "updatedAt": now,
                    }
                )
            memory["semantic"] = semantic[-100:]

        path = _memory_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")
    return record

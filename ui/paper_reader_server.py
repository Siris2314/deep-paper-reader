from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import fitz
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(REPO_ROOT / ".env")

from paper_agent.concepts import get_or_create_concept_card, save_concept_index  # noqa: E402
from paper_agent.concept_enrichment import enrich_concept_card  # noqa: E402
from paper_agent.context_diagnostics import context_diagnostics_payload  # noqa: E402
from paper_agent.judge_memory import record_math_judge_feedback  # noqa: E402
from paper_agent.math_agent import explain_equation  # noqa: E402
from paper_agent.math_regions import extract_math_regions  # noqa: E402
from paper_agent.observability import (  # noqa: E402
    capture_content,
    flush_observability,
    observability_status,
    paper_session_id,
    score_session,
    workflow_trace,
)
from paper_agent.paper_chat import (  # noqa: E402
    load_chat_thread,
    record_chat_feedback,
    run_paper_chat,
)
from paper_agent.research_lineage import build_research_lineage, research_lineage_payload  # noqa: E402
from paper_agent.parser import (  # noqa: E402
    ParsedPaper,
    clean_line,
    detect_section_heading,
    load_parsed_paper_from_workspace,
    parse_paper,
)
from paper_agent.skill_router import route_question_to_skills  # noqa: E402
from paper_agent.skills import skill_manifest  # noqa: E402
from paper_agent.term_highlighter import (  # noqa: E402
    SignificantTerm,
    load_significant_terms,
    save_significant_terms,
    term_pattern,
)
from paper_agent.term_memory import normalize_term, remember_term  # noqa: E402
from paper_agent.workspace import Workspace  # noqa: E402

UPLOAD_ROOT = REPO_ROOT / "uploaded_papers"
REPORT_ROOT = REPO_ROOT / "paper_reports"
DEFAULT_PORT = int(os.getenv("PAPER_READER_PORT", "8503"))
BUILD_LABEL = "reader-build-2026-07-28-aligned-judge-v32"
ENRICHMENT_JOB_TIMEOUT = max(30.0, min(float(os.getenv("ENRICHMENT_JOB_TIMEOUT", "240")), 600.0))
ENRICHMENT_JOBS: dict[str, dict[str, object]] = {}
ENRICHMENT_LOCK = threading.RLock()
CHAT_JOB_TIMEOUT = max(60.0, min(float(os.getenv("PAPER_CHAT_JOB_TIMEOUT", "480")), 1200.0))
CHAT_JOBS: dict[str, dict[str, object]] = {}
CHAT_LOCK = threading.RLock()


def _job_view(job: dict[str, object]) -> dict[str, object]:
    now = time.monotonic()
    with ENRICHMENT_LOCK:
        if (
            job.get("status") == "running"
            and now - float(job["started_monotonic"]) > ENRICHMENT_JOB_TIMEOUT
        ):
            job.update(
                status="failed",
                stage="timeout",
                message=(
                    "Enrichment timed out. Tavily or the local model did not respond in time; "
                    "open the term again to retry."
                ),
                updated_at=time.time(),
            )
        return {
            "jobId": job["id"],
            "status": job["status"],
            "stage": job["stage"],
            "message": job["message"],
            "elapsedSeconds": round(now - float(job["started_monotonic"]), 1),
            "result": job.get("result"),
        }


def _update_enrichment_job(job_id: str, stage: str, message: str) -> None:
    with ENRICHMENT_LOCK:
        job = ENRICHMENT_JOBS.get(job_id)
        if not job or job.get("status") != "running":
            return
        job.update(stage=stage, message=message, updated_at=time.time())


def start_enrichment_job(workspace: Workspace, parsed: ParsedPaper, term: str) -> dict[str, object]:
    key = f"{workspace.root}::{term.lower()}"
    with ENRICHMENT_LOCK:
        for existing in ENRICHMENT_JOBS.values():
            if existing.get("key") == key and existing.get("status") == "running":
                return _job_view(existing)
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "key": key,
            "status": "running",
            "stage": "queued",
            "message": "Starting concept enrichment.",
            "started_monotonic": time.monotonic(),
            "updated_at": time.time(),
            "result": None,
        }
        ENRICHMENT_JOBS[job_id] = job

    def work() -> None:
        try:
            result = enrich_term_with_tavily(
                workspace,
                parsed,
                term,
                progress=lambda stage, message: _update_enrichment_job(job_id, stage, message),
            )
        except Exception as exc:
            result = {
                "ok": False,
                "message": f"Concept enrichment stopped unexpectedly: {type(exc).__name__}: {exc}",
                "term": term,
                "sources": [],
            }
        with ENRICHMENT_LOCK:
            current = ENRICHMENT_JOBS.get(job_id)
            if not current or current.get("status") != "running":
                return
            ok = bool(result.get("ok"))
            current.update(
                status="completed" if ok else "failed",
                stage="complete" if ok else "error",
                message=str(
                    result.get("message")
                    or ("Enrichment complete." if ok else "Enrichment failed.")
                ),
                result=result,
                updated_at=time.time(),
            )

    threading.Thread(target=work, name=f"concept-enrichment-{job_id[:8]}", daemon=True).start()
    return _job_view(job)


def get_enrichment_job(job_id: str) -> dict[str, object]:
    with ENRICHMENT_LOCK:
        job = ENRICHMENT_JOBS.get(job_id)
    if not job:
        raise KeyError("Unknown enrichment job")
    return _job_view(job)


def _chat_job_view(job: dict[str, object]) -> dict[str, object]:
    now = time.monotonic()
    with CHAT_LOCK:
        if (
            job.get("status") == "running"
            and now - float(job["started_monotonic"]) > CHAT_JOB_TIMEOUT
        ):
            job.update(
                status="failed",
                stage="timeout",
                message="Paper chat timed out while waiting for a local model or web search.",
                updated_at=time.time(),
            )
        return {
            "jobId": job["id"],
            "threadId": job["thread_id"],
            "status": job["status"],
            "stage": job["stage"],
            "message": job["message"],
            "elapsedSeconds": round(now - float(job["started_monotonic"]), 1),
            "result": job.get("result"),
        }


def _update_chat_job(job_id: str, stage: str, message: str) -> None:
    with CHAT_LOCK:
        job = CHAT_JOBS.get(job_id)
        if not job or job.get("status") != "running":
            return
        job.update(stage=stage, message=message, updated_at=time.time())


def start_chat_job(
    workspace: Workspace,
    parsed: ParsedPaper,
    question: str,
    mode: str,
    thread_id: str | None,
) -> dict[str, object]:
    job_id = uuid.uuid4().hex
    effective_thread_id = str(thread_id or uuid.uuid4().hex)
    job: dict[str, object] = {
        "id": job_id,
        "thread_id": effective_thread_id,
        "status": "running",
        "stage": "queued",
        "message": "Preparing the paper evidence plan.",
        "started_monotonic": time.monotonic(),
        "updated_at": time.time(),
        "result": None,
    }
    with CHAT_LOCK:
        CHAT_JOBS[job_id] = job

    def work() -> None:
        try:
            result = run_paper_chat(
                parsed,
                workspace,
                question,
                mode=mode,  # type: ignore[arg-type]
                thread_id=effective_thread_id,
                progress=lambda stage, message: _update_chat_job(job_id, stage, message),
            )
            with CHAT_LOCK:
                current = CHAT_JOBS.get(job_id)
                if not current or current.get("status") != "running":
                    return
                current.update(
                    status="completed",
                    stage="complete",
                    message="Paper chat answer is ready.",
                    result=result.model_dump(),
                    updated_at=time.time(),
                )
        except Exception as exc:
            with CHAT_LOCK:
                current = CHAT_JOBS.get(job_id)
                if not current or current.get("status") != "running":
                    return
                current.update(
                    status="failed",
                    stage="error",
                    message=f"Paper chat failed: {exc}",
                    updated_at=time.time(),
                )

    threading.Thread(target=work, name=f"paper-chat-{job_id[:8]}", daemon=True).start()
    return _chat_job_view(job)


def get_chat_job(job_id: str) -> dict[str, object]:
    with CHAT_LOCK:
        job = CHAT_JOBS.get(job_id)
    if not job:
        raise KeyError("Unknown paper chat job")
    return _chat_job_view(job)


def start_math_job(
    workspace: Workspace,
    parsed: ParsedPaper,
    page: int,
    equation_id: str,
    raw: str,
    context: str,
    region_kind: str,
    layout_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    layout_key = json.dumps(layout_evidence or {}, sort_keys=True)
    key = f"{workspace.root}::math::{page}::{hashlib.sha256((region_kind + raw + context + layout_key).encode('utf-8')).hexdigest()[:16]}"
    with ENRICHMENT_LOCK:
        for existing in ENRICHMENT_JOBS.values():
            if existing.get("key") == key and existing.get("status") == "running":
                return _job_view(existing)
        job_id = uuid.uuid4().hex
        job: dict[str, object] = {
            "id": job_id,
            "key": key,
            "status": "running",
            "stage": "queued",
            "message": "Starting math analysis.",
            "started_monotonic": time.monotonic(),
            "updated_at": time.time(),
            "result": None,
        }
        ENRICHMENT_JOBS[job_id] = job

    def work() -> None:
        try:
            explanation = explain_equation(
                parsed,
                workspace,
                page,
                equation_id,
                raw,
                context=context,
                region_kind=region_kind,
                layout_evidence=layout_evidence,
                progress=lambda stage, message: _update_enrichment_job(job_id, stage, message),
            )
            result: dict[str, object] = {
                "ok": True,
                "message": "Math explanation complete.",
                "math": asdict(explanation),
            }
        except Exception as exc:
            result = {"ok": False, "message": f"Math explanation failed: {exc}"}
        with ENRICHMENT_LOCK:
            current = ENRICHMENT_JOBS.get(job_id)
            if not current or current.get("status") != "running":
                return
            ok = bool(result.get("ok"))
            current.update(
                status="completed" if ok else "failed",
                stage="complete" if ok else "error",
                message=str(result.get("message")),
                result=result,
                updated_at=time.time(),
            )

    threading.Thread(target=work, name=f"math-explanation-{job_id[:8]}", daemon=True).start()
    return _job_view(job)


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    return "-".join(part for part in cleaned.split("-") if part)[:90] or "paper"


def short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


def workspace_for_pdf(pdf_path: Path) -> Path:
    return REPORT_ROOT / f"{slugify(pdf_path.stem)}-{short_hash(pdf_path.read_bytes())}"


def latest_workspace() -> Path | None:
    candidates: list[Path] = []
    for root in [REPORT_ROOT, REPO_ROOT]:
        if not root.exists():
            continue
        candidates.extend(
            [p for p in root.iterdir() if p.is_dir() and (p / "parsed" / "metadata.json").exists()]
        )
    flat = REPO_ROOT / "paper_report"
    if (flat / "parsed" / "metadata.json").exists():
        candidates.append(flat)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "parsed" / "metadata.json").stat().st_mtime)


def parse_uploaded_pdf(filename: str, data: bytes) -> tuple[Workspace, ParsedPaper]:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = f"{slugify(Path(filename).stem)}-{short_hash(data)}.pdf"
    pdf_path = UPLOAD_ROOT / safe_name
    if not pdf_path.exists() or pdf_path.read_bytes() != data:
        pdf_path.write_bytes(data)
    workspace = Workspace(workspace_for_pdf(pdf_path))
    with workflow_trace(
        "paper-upload-and-parse",
        input_data={"filename": filename, "pdf": data},
        session_id=paper_session_id(workspace),
        tags=["parser", "paper-reader"],
        metadata={"pdfBytes": len(data), "pdfHash": short_hash(data)},
        as_type="chain",
    ) as trace:
        workspace.reset()
        parsed = parse_paper(pdf_path, workspace.root)
        save_concept_index(parsed, workspace)
        terms = save_significant_terms(parsed, workspace)
        workspace.write_text("logs/active_pdf.txt", f"source_pdf={pdf_path}\n")
        trace.update(
            output={
                "pages": parsed.metadata.page_count,
                "equations": len(parsed.equation_cards),
                "figures": len(parsed.figure_cards),
                "tables": len(parsed.table_cards),
                "significantTerms": len(terms),
            }
        )
        trace.score("parse_succeeded", 1.0, data_type="BOOLEAN")
        trace.score("parse_pages", float(parsed.metadata.page_count))
        return workspace, parsed


def load_workspace(workspace_arg: str | None = None) -> tuple[Workspace, ParsedPaper]:
    root = (
        Path(unquote(workspace_arg)).expanduser().resolve() if workspace_arg else latest_workspace()
    )
    if root is None:
        raise FileNotFoundError("No parsed workspace found. Upload a PDF first.")
    workspace = Workspace(root)
    return workspace, load_parsed_paper_from_workspace(workspace)


def matching_equations(parsed: ParsedPaper, term: str) -> list[dict[str, object]]:
    low = term.lower()
    return [card.model_dump() for card in parsed.equation_cards if low in card.raw.lower()][:12]


def matching_visuals(parsed: ParsedPaper, term: str) -> dict[str, list[dict[str, object]]]:
    low = term.lower()
    figures = [
        card.model_dump()
        for card in parsed.figure_cards
        if low in (card.caption + "\n" + card.surrounding_text).lower()
    ][:8]
    tables = [
        card.model_dump()
        for card in parsed.table_cards
        if low in (card.caption + "\n" + card.raw_text).lower()
    ][:8]
    return {"figures": figures, "tables": tables}


def source_pdf_path(parsed: ParsedPaper) -> Path:
    source = Path(parsed.metadata.source_pdf or "").expanduser().resolve()
    if not source.exists() or source.suffix.lower() != ".pdf":
        raise FileNotFoundError(
            "The original PDF for this workspace is unavailable. Upload it again."
        )
    return source


def pdf_page_sizes(parsed: ParsedPaper) -> list[dict[str, float | int]]:
    with fitz.open(source_pdf_path(parsed)) as doc:
        return [
            {
                "page": index + 1,
                "width": round(page.rect.width, 3),
                "height": round(page.rect.height, 3),
            }
            for index, page in enumerate(doc)
        ]


def section_navigation(parsed: ParsedPaper) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    seen: set[str] = set()
    for page_number, text in sorted(parsed.page_text.items()):
        for line in text.splitlines():
            label = clean_line(line)
            if not detect_section_heading(label):
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            sections.append({"label": label, "page": page_number})
    return sections[:40]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _automatic_margin_annotation(
    matched_words: list[dict[str, object]], page_width: float, page_height: float
) -> bool:
    if not matched_words:
        return False
    x0 = min(float(word["x"]) for word in matched_words)
    y0 = min(float(word["y"]) for word in matched_words)
    x1 = max(float(word["x"]) + float(word["width"]) for word in matched_words)
    y1 = max(float(word["y"]) + float(word["height"]) for word in matched_words)
    side_margin = x1 <= page_width * 0.10 or x0 >= page_width * 0.90
    edge_header = y1 <= page_height * 0.025 or y0 >= page_height * 0.975
    vertical_text = (y1 - y0) > max((x1 - x0) * 1.8, 20.0)
    return edge_header or (side_margin and vertical_text)


def page_layout_payload(
    parsed: ParsedPaper, terms: list[SignificantTerm], page_number: int
) -> dict[str, object]:
    with fitz.open(source_pdf_path(parsed)) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page {page_number} is outside this PDF.")
        page = doc[page_number - 1]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        raw_words = page.get_text("words", sort=True)
        page_dict = page.get_text("dict", sort=True)

    words = [
        {
            "x": round(float(item[0]), 3),
            "y": round(float(item[1]), 3),
            "width": round(float(item[2] - item[0]), 3),
            "height": round(float(item[3] - item[1]), 3),
            "text": str(item[4]),
            "block": int(item[5]),
            "line": int(item[6]),
            "word": int(item[7]),
        }
        for item in raw_words
        if str(item[4]).strip()
    ]
    stream: list[tuple[str, int]] = []
    for word_index, word in enumerate(words):
        stream.extend((token, word_index) for token in _tokens(str(word["text"])))

    highlights: list[dict[str, object]] = []
    claimed_words: set[int] = set()
    ordered_terms = sorted(
        terms,
        key=lambda item: (item.learned, len(_tokens(item.term)), len(item.term), item.score),
        reverse=True,
    )
    for term in ordered_terms:
        wanted = _tokens(term.term)
        if not wanted:
            continue
        for start in range(0, len(stream) - len(wanted) + 1):
            if [token for token, _ in stream[start : start + len(wanted)]] != wanted:
                continue
            word_ids = list(
                dict.fromkeys(index for _, index in stream[start : start + len(wanted)])
            )
            if any(index in claimed_words for index in word_ids):
                continue
            matched_words = [words[index] for index in word_ids]
            if not term.learned and _automatic_margin_annotation(
                matched_words, page_width, page_height
            ):
                continue
            claimed_words.update(word_ids)
            grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
            for index in word_ids:
                word = words[index]
                grouped.setdefault((int(word["block"]), int(word["line"])), []).append(word)
            for line_words in grouped.values():
                x0 = min(float(word["x"]) for word in line_words)
                y0 = min(float(word["y"]) for word in line_words)
                x1 = max(float(word["x"]) + float(word["width"]) for word in line_words)
                y1 = max(float(word["y"]) + float(word["height"]) for word in line_words)
                highlights.append(
                    {
                        "x": round(x0, 3),
                        "y": round(y0, 3),
                        "width": round(x1 - x0, 3),
                        "height": round(y1 - y0, 3),
                        "term": term.term,
                        "category": term.category,
                        "learned": term.learned,
                    }
                )

    math_regions = extract_math_regions(page_dict, page_number, page_width, page_height)

    return {
        "page": page_number,
        "width": round(page_width, 3),
        "height": round(page_height, 3),
        "words": words,
        "highlights": highlights,
        "mathRegions": math_regions,
    }


def render_page_png(workspace: Workspace, parsed: ParsedPaper, page_number: int) -> bytes:
    scale = max(1.0, min(float(os.getenv("PAPER_RENDER_SCALE", "1.7")), 3.0))
    cache_path = workspace.path(f"parsed/page_renders/{page_number:04d}-{scale:.2f}.png")
    if cache_path.exists():
        return cache_path.read_bytes()
    with fitz.open(source_pdf_path(parsed)) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page {page_number} is outside this PDF.")
        pixmap = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        content = pixmap.tobytes("png")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def remember_highlight(
    workspace: Workspace,
    parsed: ParsedPaper,
    term: str,
    category: str,
    page: int | None,
) -> dict[str, object]:
    clean_term = normalize_term(term)
    if not re.search(term_pattern(clean_term), parsed.full_text, re.I):
        raise ValueError(
            "That phrase was not found in the parsed paper text. Select or enter it exactly."
        )
    record = remember_term(
        clean_term,
        category,
        paper_title=parsed.metadata.title_guess,
        source_pdf=parsed.metadata.source_pdf,
        page=page,
    )
    workspace.write_json(
        "memory/manual_highlights.json",
        {
            "description": "User-confirmed terms for this paper. Shared learning lives in agent_memory/term_highlights.json.",
            "terms": [
                asdict(term_item)
                for term_item in save_significant_terms(parsed, workspace, max_terms=160)
                if term_item.learned
            ],
        },
    )
    return {"ok": True, "memory": record, "state": state_payload(workspace, parsed)}


def state_payload(workspace: Workspace, parsed: ParsedPaper) -> dict[str, object]:
    terms = load_significant_terms(parsed, workspace)
    return {
        "workspace": str(workspace.root),
        "metadata": parsed.metadata.model_dump(),
        "pages": pdf_page_sizes(parsed),
        "sections": section_navigation(parsed),
        "terms": [asdict(term) for term in terms],
        "skills": skill_manifest(),
        "equationCount": len(parsed.equation_cards),
        "figureCount": len(parsed.figure_cards),
        "tableCount": len(parsed.table_cards),
        "tavilyReady": bool(os.getenv("TAVILY_API_KEY")),
        "observability": observability_status(),
        "chat": {
            "ready": True,
            "modes": ["fast", "deep", "web", "explore"],
            "model": os.getenv("PAPER_READER_AGENT_MODEL", os.getenv("MODEL_NAME", "qwen2.5:7b")),
            "mathModel": os.getenv("PAPER_READER_MATH_MODEL", "qwen3.5:4b"),
        },
        "build": BUILD_LABEL,
    }


def term_payload(workspace: Workspace, parsed: ParsedPaper, term: str) -> dict[str, object]:
    card = get_or_create_concept_card(parsed, workspace, term)
    terms = load_significant_terms(parsed, workspace)
    sig = next((item for item in terms if item.term.lower() == term.lower()), None)
    equations = matching_equations(parsed, term)
    visuals = matching_visuals(parsed, term)
    skill_names = ["concept-card", "research-lineage"]
    if equations:
        skill_names.append("math-walkthrough")
    if visuals["figures"] or visuals["tables"]:
        skill_names.append("figure-table-analysis")
    trace_by_name = {
        trigger.name: asdict(trigger)
        for trigger in route_question_to_skills(
            "Define this concept and trace its prior work."
            + (" Explain its equation." if equations else "")
            + (" Inspect its figure or table." if visuals["figures"] or visuals["tables"] else "")
        )
    }
    return {
        "term": term,
        "card": asdict(card),
        "significantTerm": asdict(sig) if sig else None,
        "equations": equations,
        "visuals": visuals,
        "skillTrace": [trace_by_name[name] for name in skill_names if name in trace_by_name],
        "tavilyReady": bool(os.getenv("TAVILY_API_KEY")),
    }


def enrich_term_with_tavily(
    workspace: Workspace,
    parsed: ParsedPaper,
    term: str,
    progress: object | None = None,
) -> dict[str, object]:
    if not os.getenv("TAVILY_API_KEY"):
        return {
            "ok": False,
            "message": "TAVILY_API_KEY is not set. Add it to .env to enable lazy web enrichment.",
            "sources": [],
        }
    existing_card = get_or_create_concept_card(parsed, workspace, term)
    if (
        existing_card.general_explanation_source in {"tavily_research", "tavily_search_answer"}
        and existing_card.why_it_matters_source == "paper_agent"
    ):
        return {
            "ok": True,
            "message": "Loaded cached Tavily research and paper-agent analysis.",
            "sources": existing_card.web_sources,
            "term": term,
            "card": asdict(existing_card),
        }

    try:
        card = enrich_concept_card(
            parsed, workspace, term, progress=progress if callable(progress) else None
        )
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Concept enrichment failed: {exc}",
            "sources": [],
            "term": term,
        }

    # Keep the flat source cache for other agent and debug surfaces.
    sources_path = workspace.path("web/tavily_sources.json")
    try:
        existing = (
            json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else []
        )
    except Exception:
        existing = []
    seen = {
        str(source.get("url") or source.get("title") or "").lower()
        for source in existing
        if isinstance(source, dict)
    }
    merged = list(existing)
    for source in card.web_sources:
        key = str(source.get("url") or source.get("title") or "").lower()
        if key and key not in seen:
            seen.add(key)
            merged.append({**source, "term": term, "provider": card.general_explanation_source})
    workspace.write_json("web/tavily_sources.json", merged)

    complete = card.why_it_matters_source == "paper_agent"
    return {
        "ok": complete,
        "message": (
            "Tavily research and paper-agent analysis complete."
            if complete
            else f"Tavily research complete, but the paper agent failed: {card.why_it_matters_here}"
        ),
        "sources": card.web_sources,
        "term": term,
        "card": asdict(card),
    }


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep Paper Reader</title>
  <script>
    window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #182230;
      --muted: #667085;
      --line: #d6dde6;
      --math: #dff4ff;
      --method: #f3e8ff;
      --result: #e7f8e8;
      --concept: #fff2a8;
      --accent: #b8322a;
      --accent-dark: #922820;
      --sidebar: #eef1f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    .shell { display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: 100vh; }
    aside {
      background: var(--sidebar);
      border-right: 1px solid var(--line);
      padding: 18px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
    }
    main { padding: 18px 24px 30px; min-width: 0; }
    h1 { margin: 0; font-size: 24px; line-height: 1.2; }
    h2 { font-size: 14px; margin: 18px 0 8px; }
    .subtle { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .brand { display: flex; align-items: center; gap: 9px; margin-bottom: 18px; }
    .brand-mark {
      width: 30px;
      height: 30px;
      display: grid;
      place-items: center;
      border-radius: 7px;
      background: var(--ink);
      color: #fff;
      font-family: Georgia, "Times New Roman", serif;
      font-weight: 800;
      font-size: 17px;
    }
    .brand-name { font-weight: 760; font-size: 15px; }
    .brand-version { color: var(--muted); font-size: 10px; }
    .sidebar-label {
      color: #475467;
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
      margin: 16px 0 7px;
    }
    .upload {
      border: 1px dashed #aab4c4;
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }
    input[type=file] { width: 100%; font-size: 12px; }
    button {
      border: 1px solid #cfd7e3;
      background: #fff;
      color: var(--ink);
      padding: 9px 11px;
      border-radius: 7px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; width: 100%; margin-top: 10px; }
    button.primary:hover { background: var(--accent-dark); border-color: var(--accent-dark); }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .loading-overlay {
      position: fixed;
      inset: 0;
      background: rgba(245, 247, 251, .84);
      backdrop-filter: blur(3px);
      z-index: 80;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .loading-overlay.open { display: flex; }
    .loading-panel {
      width: min(520px, calc(100vw - 44px));
      background: #fff;
      border: 1px solid #cbd4e1;
      border-radius: 10px;
      box-shadow: 0 24px 70px rgba(15, 23, 42, .2);
      padding: 22px 24px;
    }
    .spinner {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      border: 4px solid #e4e9f2;
      border-top-color: var(--accent);
      animation: spin .8s linear infinite;
      margin-bottom: 14px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-panel h3 { margin: 0 0 6px; font-size: 20px; }
    .loading-detail { color: var(--muted); font-size: 14px; line-height: 1.5; margin-bottom: 14px; }
    .progress-track { height: 8px; background: #edf1f7; border-radius: 999px; overflow: hidden; }
    .progress-bar {
      height: 100%;
      width: 18%;
      border-radius: 999px;
      background: var(--accent);
      transition: width .35s ease;
    }
    .reader.busy {
      opacity: .62;
    }
    .upload.busy {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(192, 53, 43, .1);
    }
    .app-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-bottom: 12px; }
    .app-heading { min-width: 0; }
    .app-heading h1 { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .paper-meta { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .stat-grid { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
    .stat {
      display: flex;
      align-items: baseline;
      gap: 4px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 5px 8px;
    }
    .stat b { font-size: 13px; }
    .stat span { color: var(--muted); font-size: 11px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 430px; gap: 18px; align-items: start; }
    .reader {
      height: calc(100vh - 76px);
      overflow-y: auto;
      background: #dfe4ec;
      border: 1px solid #cbd4e1;
      border-radius: 9px;
      padding: 20px;
    }
    .page {
      max-width: 900px;
      margin: 0 auto 18px;
      background: #fff;
      border: 1px solid #ccd4df;
      border-radius: 5px;
      box-shadow: 0 12px 26px rgba(15, 23, 42, .08);
      padding: 42px 48px;
      line-height: 1.68;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 16px;
      overflow-wrap: break-word;
    }
    .page-number {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 18px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .paper-title {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 28px;
      line-height: 1.18;
      margin: 0 0 18px;
      font-weight: 780;
      color: #111827;
    }
    .section-heading {
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-size: 20px;
      line-height: 1.2;
      margin: 24px 0 10px;
      padding-top: 10px;
      border-top: 1px solid #e2e8f0;
      font-weight: 760;
      color: #111827;
    }
    .section-heading.abstract {
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #344054;
      border-top: 0;
      margin-top: 10px;
    }
    .para { margin: 0 0 12px; }
    .math-line {
      font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      background: #f7fbff;
      border: 1px solid #cce7f6;
      border-left: 3px solid #58a6c8;
      border-radius: 6px;
      padding: 10px 12px;
      margin: 12px 0 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      line-height: 1.55;
      color: #0f3b51;
      font-size: 14px;
    }
    .term {
      border-radius: 4px;
      padding: 0 2px;
      cursor: pointer;
      border: 1px solid transparent;
      font-weight: 650;
    }
    .term.math { background: var(--math); border-color: #89cbe6; }
    .term.method { background: var(--method); border-color: #c9a7f1; }
    .term.result { background: var(--result); border-color: #91cd96; }
    .term.concept { background: var(--concept); border-color: #ddca59; }
    .side {
      position: sticky;
      top: 20px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      max-height: calc(100vh - 40px);
    }
    .side-tabs { display: grid; grid-template-columns: 1fr 1fr; border: 1px solid var(--line); border-radius: 7px; background: #e9edf3; padding: 3px; }
    .side-tab { border: 0; background: transparent; padding: 8px; color: #526178; }
    .side-tab.active { background: #fff; color: var(--ink); box-shadow: 0 1px 3px rgba(15, 23, 42, .12); }
    .side-panel { min-height: 0; }
    .side-panel[hidden] { display: none; }
    .inspect-panel { overflow-y: auto; display: grid; gap: 12px; padding-right: 2px; }
    .chat-panel {
      height: calc(100vh - 92px);
      min-height: 520px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .chat-header { display: flex; align-items: center; gap: 9px; padding: 12px 13px; border-bottom: 1px solid var(--line); }
    .chat-header h2 { margin: 0; font-size: 15px; }
    .chat-model { color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: auto; }
    .chat-icon-button { width: 30px; height: 30px; padding: 0; display: grid; place-items: center; font-size: 17px; }
    .chat-messages { min-height: 0; overflow-y: auto; padding: 14px; background: #fafbfd; }
    .chat-empty { color: var(--muted); font-size: 13px; line-height: 1.5; padding: 16px 4px; }
    .chat-message { margin-bottom: 14px; }
    .chat-message.user { display: flex; justify-content: flex-end; }
    .chat-bubble { max-width: 92%; border-radius: 7px; padding: 10px 11px; font-size: 14px; line-height: 1.52; overflow-wrap: anywhere; }
    .chat-message.user .chat-bubble { background: #24324a; color: #fff; }
    .chat-message.assistant .chat-bubble { max-width: 100%; padding: 0; }
    .chat-answer p { margin: 0 0 9px; }
    .chat-answer p:last-child { margin-bottom: 0; }
    .chat-answer ul, .chat-answer ol { margin: 6px 0 10px; padding-left: 22px; }
    .chat-answer li { margin-bottom: 4px; }
    .chat-answer code { font: 12px/1.45 "Cascadia Mono", Consolas, monospace; background: #edf1f6; padding: 1px 4px; border-radius: 3px; }
    .chat-answer pre { overflow-x: auto; background: #edf1f6; border: 1px solid #d6dde7; border-radius: 6px; padding: 9px; }
    .chat-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-top: 8px; color: var(--muted); font-size: 11px; }
    .chat-route { border: 1px solid #d6dde7; border-radius: 999px; padding: 2px 6px; background: #fff; }
    .chat-citations { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
    .citation-button { padding: 4px 7px; border-radius: 5px; font-size: 11px; color: #184f8a; background: #edf6ff; border-color: #b8d5ef; }
    .citation-button.web { color: #17612b; background: #edf8ef; border-color: #b9dabf; }
    .chat-trace { margin-top: 9px; border-top: 1px solid #e1e6ee; padding-top: 7px; }
    .chat-trace summary { cursor: pointer; color: #526178; font-size: 11px; font-weight: 700; }
    .trace-list { display: grid; gap: 5px; margin-top: 7px; color: #526178; font-size: 11px; }
    .trace-item { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: 6px; }
    .chat-feedback { display: flex; gap: 5px; margin-top: 8px; }
    .chat-feedback button { width: 28px; height: 25px; padding: 0; font-size: 13px; }
    .chat-correction { display: none; gap: 7px; margin-top: 8px; padding: 9px; border: 1px solid #d5dce6; border-radius: 6px; background: #fff; }
    .chat-correction.open { display: grid; }
    .chat-correction textarea { min-height: 72px; resize: vertical; }
    .memory-check { display: flex; align-items: center; gap: 7px; color: #475467; font-size: 12px; }
    .chat-composer { border-top: 1px solid var(--line); padding: 10px; background: #fff; }
    .mode-control { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3px; background: #edf1f6; border-radius: 6px; padding: 3px; margin-bottom: 8px; }
    .mode-control button { border: 0; background: transparent; color: #5a6678; padding: 6px 3px; font-size: 11px; border-radius: 4px; }
    .mode-control button.active { background: #fff; color: #172033; box-shadow: 0 1px 3px rgba(15, 23, 42, .12); }
    .composer-row { display: grid; grid-template-columns: minmax(0, 1fr) 38px; gap: 7px; align-items: end; }
    .composer-row textarea, .chat-correction textarea {
      width: 100%; border: 1px solid #cfd7e3; border-radius: 6px; background: #fff; color: var(--ink);
      padding: 9px 10px; font: inherit; line-height: 1.4;
    }
    .composer-row textarea { min-height: 62px; max-height: 160px; resize: vertical; }
    .send-button { width: 38px; height: 38px; padding: 0; display: grid; place-items: center; background: var(--accent); border-color: var(--accent); color: #fff; font-size: 19px; }
    .chat-status { min-height: 17px; margin-top: 6px; color: #526178; font-size: 11px; }
    .chat-thinking { display: flex; align-items: center; gap: 8px; color: #526178; font-size: 13px; }
    .chat-thinking .spinner { width: 18px; height: 18px; border-width: 2px; margin: 0; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .card h2 { margin-top: 0; }
    .term-list { display: flex; flex-wrap: wrap; gap: 6px; max-height: 250px; overflow-y: auto; }
    .pill {
      border: 1px solid #d6dde7;
      border-radius: 999px;
      background: #fff;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
    }
    .pill.math { background: var(--math); border-color: #89cbe6; }
    .pill.method { background: var(--method); border-color: #c9a7f1; }
    .pill.result { background: var(--result); border-color: #91cd96; }
    .pill.concept { background: var(--concept); border-color: #ddca59; }
    .popover {
      position: fixed;
      top: 18px;
      left: 50%;
      transform: translateX(-50%) translateY(-18px);
      width: min(820px, calc(100vw - 32px));
      max-height: min(84vh, 780px);
      overflow-y: auto;
      background: #fff;
      border: 1px solid #b7c1d0;
      border-radius: 10px;
      box-shadow: 0 24px 70px rgba(15, 23, 42, .26);
      padding: 18px 20px 20px;
      z-index: 50;
      opacity: 0;
      pointer-events: none;
      transition: .16s ease;
    }
    .popover.open { opacity: 1; pointer-events: auto; transform: translateX(-50%) translateY(0); }
    .popover-header {
      position: sticky;
      top: -18px;
      z-index: 3;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding: 16px 20px 10px;
      margin: -18px -20px 12px;
      background: rgba(255, 255, 255, .97);
      backdrop-filter: blur(8px);
    }
    .popover h3 { margin: 0; font-size: 22px; }
    .popover h4 { margin: 13px 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #475467; }
    .popover p { margin: 5px 0 9px; line-height: 1.5; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .tag { border: 1px solid #d5dce8; background: #f7f9fc; padding: 3px 7px; border-radius: 999px; font-size: 12px; }
    .popover-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
    .enrich-status { color: var(--muted); font-size: 13px; }
    .concept-meta { display: flex; flex-wrap: wrap; gap: 5px 12px; color: var(--muted); font-size: 12px; }
    .concept-status { padding: 8px 0 10px; border-bottom: 1px solid var(--line); }
    .concept-section { padding: 14px 0 12px; border-bottom: 1px solid var(--line); }
    .concept-section:last-child { border-bottom: 0; }
    .concept-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 7px; }
    .concept-section-heading h4 { margin: 0; font-size: 13px; text-transform: none; letter-spacing: 0; color: #344054; }
    .provenance { border: 1px solid #d5dce8; background: #f7f9fc; color: #596579; padding: 2px 6px; border-radius: 4px; font-size: 10px; white-space: nowrap; }
    .concept-copy { color: #202b3d; font-size: 15px; line-height: 1.62; }
    .concept-copy p { margin: 0 0 9px; line-height: inherit; }
    .concept-copy p:last-child { margin-bottom: 0; }
    .concept-details { padding: 11px 0; border-bottom: 1px solid var(--line); }
    .concept-details > summary { cursor: pointer; color: #344054; font-size: 13px; font-weight: 700; }
    .concept-details[open] > summary { margin-bottom: 9px; }
    .inline-tech-math, .inline-math-source { white-space: nowrap; }
    .inline-math-fallback {
      display: inline;
      border: 0;
      border-radius: 3px;
      background: #f2f4f7;
      color: #344054;
      padding: 1px 3px;
      font: 0.92em/1.35 "Cascadia Mono", Consolas, monospace;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .section-list { display: grid; gap: 5px; max-height: 210px; overflow-y: auto; }
    .section-link {
      text-align: left;
      border: 0;
      border-radius: 6px;
      background: #f7f9fc;
      color: #24324a;
      font-weight: 620;
      padding: 7px 8px;
      font-size: 13px;
    }
    .evidence { padding-left: 20px; margin: 6px 0 10px; }
    .evidence li { padding-left: 2px; margin: 0 0 5px; line-height: 1.48; }
    .equation-card {
      font-family: "Cascadia Mono", Consolas, monospace;
      border: 1px solid #cce7f6;
      background: #f7fbff;
      border-radius: 6px;
      padding: 9px;
      overflow-x: auto;
      margin: 7px 0;
      white-space: pre-wrap;
    }
    .math-rendered {
      min-height: 74px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow-x: auto;
      background: #fbfdff;
      border: 1px solid #cce0ee;
      border-radius: 7px;
      padding: 16px 18px;
      margin: 8px 0;
      font-size: 18px;
    }
    .math-render-fallback {
      min-height: 74px;
      display: grid;
      place-items: center;
      border: 1px solid #d8dee8;
      border-radius: 7px;
      background: #fafbfd;
      color: #475467;
      padding: 16px 18px;
      text-align: center;
      font-size: 13px;
      line-height: 1.5;
    }
    .trace-details { border-top: 1px solid var(--line); margin-top: 10px; padding-top: 8px; }
    .trace-details summary { color: #475467; cursor: pointer; font-size: 13px; font-weight: 650; }
    .technical-details { border: 1px solid var(--line); border-radius: 7px; margin: 14px 0; padding: 9px 11px; background: #fafbfd; }
    .technical-details > summary { cursor: pointer; color: #344054; font-size: 13px; font-weight: 700; }
    .technical-details[open] > summary { margin-bottom: 8px; }
    .implementation-view { margin: 6px 0 10px; }
    .implementation-view p { margin: 0 0 8px; }
    .implementation-code {
      margin: 7px 0 10px;
      padding: 11px 12px;
      overflow-x: auto;
      border: 1px solid #cbd7e5;
      border-radius: 6px;
      background: #f6f8fb;
      color: #172033;
      font: 13px/1.48 "Cascadia Mono", Consolas, monospace;
      white-space: pre;
    }
    .math-symbol { white-space: nowrap; font-size: 15px; }
    .symbol-source { color: #475467; font-size: 12px; text-transform: capitalize; }
    .role-copy { color: #24324a; font-weight: 650; text-transform: none; }
    .paper-evidence { list-style: none; padding-left: 0; }
    .paper-evidence li {
      border-left: 2px solid #b8c4d4;
      padding: 2px 0 2px 10px;
      margin-bottom: 9px;
      color: #344054;
    }
    .judge-details > summary { display: flex; align-items: center; gap: 8px; }
    .judge-panel { border: 1px solid #cfd8e5; border-radius: 7px; padding: 12px; background: #f9fbfd; }
    .judge-header { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .judge-verdict { border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 750; text-transform: uppercase; }
    .judge-verdict.pass { color: #17612b; background: #e2f6e7; border: 1px solid #8bc89a; }
    .judge-verdict.review { color: #805400; background: #fff4d6; border: 1px solid #dfbd62; }
    .judge-verdict.fail { color: #8d2119; background: #fde8e6; border: 1px solid #dda09b; }
    .judge-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin: 10px 0; }
    .judge-score { border: 1px solid #d7dee8; background: #fff; border-radius: 6px; padding: 7px; min-width: 0; }
    .judge-score b { display: block; font-size: 18px; }
    .judge-score span { display: block; color: #667085; font-size: 10px; overflow-wrap: anywhere; }
    .judge-reasons { display: grid; gap: 7px; margin-top: 9px; }
    .judge-reason { font-size: 13px; border-left: 2px solid #aeb9c9; padding-left: 8px; }
    .judge-feedback { border-top: 1px solid #d7dee8; margin-top: 12px; padding-top: 10px; }
    .judge-feedback-actions { display: flex; gap: 7px; flex-wrap: wrap; }
    .judge-correction { display: none; gap: 7px; margin-top: 9px; }
    .judge-correction.open { display: grid; }
    .judge-correction textarea { min-height: 82px; resize: vertical; }
    .judge-correction select { max-width: 240px; }
    .source-card {
      border: 1px solid #d8dee8;
      background: #fafbfd;
      border-radius: 7px;
      padding: 9px 10px;
      margin: 8px 0;
      font-size: 14px;
    }
    .source-card a { color: #1659a7; text-decoration: none; font-weight: 650; }
    .lineage-list { display: grid; gap: 8px; }
    .lineage-card {
      border: 1px solid #d8dee8;
      border-left: 3px solid #4c78a8;
      background: #fafbfd;
      border-radius: 6px;
      padding: 10px 11px;
    }
    .lineage-card.extends, .lineage-card.adopts, .lineage-card.modifies { border-left-color: #20856b; }
    .lineage-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .lineage-title { font-weight: 700; line-height: 1.3; }
    .relation-badge { font-size: 11px; text-transform: uppercase; white-space: nowrap; }
    .citation-context { margin-top: 7px; font-size: 13px; color: #34445a; line-height: 1.45; }
    .enrichment-placeholder {
      border-left: 3px solid #9aa9bc;
      background: #f6f8fb;
      padding: 9px 11px;
      color: var(--muted);
      font-size: 13px;
    }
    .pdf-page-shell {
      width: min(100%, 980px);
      margin: 0 auto 20px;
      scroll-margin-top: 16px;
    }
    .pdf-page-label {
      color: #556176;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      margin: 0 0 6px 2px;
    }
    .pdf-page {
      position: relative;
      width: 100%;
      background: #fff;
      border: 1px solid #c7cfdb;
      box-shadow: 0 12px 26px rgba(15, 23, 42, .11);
      overflow: hidden;
      container-type: inline-size;
    }
    .pdf-page img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }
    .highlight-layer, .text-layer, .math-layer { position: absolute; inset: 0; }
    .highlight-layer { z-index: 3; pointer-events: none; }
    .text-layer { z-index: 2; overflow: hidden; user-select: text; }
    .math-layer { z-index: 4; pointer-events: none; }
    .math-hotspot {
      position: absolute;
      display: block;
      padding: 0;
      margin: 0;
      min-width: 8px;
      background: rgba(41, 146, 184, .04);
      border: 1px solid transparent;
      border-radius: 3px;
      pointer-events: auto;
      cursor: help;
    }
    .math-hotspot:hover, .math-hotspot:focus-visible {
      background: rgba(77, 184, 224, .18);
      border-color: rgba(31, 126, 164, .75);
      box-shadow: 0 0 0 2px rgba(77, 184, 224, .14);
    }
    .math-hotspot.inline { border-bottom-color: rgba(31, 126, 164, .45); }
    .math-hotspot.display { background: rgba(41, 146, 184, .07); }
    .symbol-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .symbol-table th, .symbol-table td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }
    .symbol-table th { color: #475467; font-size: 12px; text-transform: uppercase; }
    .pdf-highlight {
      position: absolute;
      display: block;
      padding: 0;
      margin: 0;
      min-width: 2px;
      border-radius: 2px;
      cursor: pointer;
      pointer-events: auto;
      mix-blend-mode: multiply;
    }
    .pdf-highlight.math { background: rgba(111, 207, 242, .32); border: 1px solid rgba(41, 146, 184, .72); }
    .pdf-highlight.method { background: rgba(190, 145, 242, .27); border: 1px solid rgba(133, 78, 188, .68); }
    .pdf-highlight.result { background: rgba(111, 207, 151, .29); border: 1px solid rgba(48, 150, 90, .68); }
    .pdf-highlight.concept { background: rgba(255, 224, 74, .34); border: 1px solid rgba(194, 154, 0, .68); }
    .pdf-highlight.learned { outline: 2px solid rgba(192, 53, 43, .62); outline-offset: 1px; }
    .pdf-highlight:hover { filter: saturate(1.35); box-shadow: 0 0 0 2px rgba(17, 24, 39, .18); }
    .text-word {
      position: absolute;
      color: transparent;
      white-space: pre;
      line-height: 1;
      overflow: hidden;
      font-family: serif;
      letter-spacing: 0;
      user-select: text;
    }
    .text-word::selection { background: rgba(28, 112, 214, .38); color: transparent; }
    .memory-form { display: grid; gap: 8px; }
    .memory-form input, .memory-form select {
      width: 100%;
      border: 1px solid #cfd7e3;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
    }
    .memory-form button { width: 100%; }
    .selection-action {
      position: fixed;
      z-index: 45;
      display: none;
      width: min(340px, calc(100vw - 24px));
      background: #fff;
      border: 1px solid #aeb9c9;
      border-radius: 8px;
      box-shadow: 0 14px 36px rgba(15, 23, 42, .24);
      padding: 10px;
    }
    .selection-action.open { display: grid; grid-template-columns: minmax(0, 1fr) 100px; gap: 8px; align-items: center; }
    .selection-action .selection-label { grid-column: 1 / -1; font-size: 12px; color: #475467; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .selection-action select { min-width: 0; border: 1px solid #cfd7e3; border-radius: 6px; padding: 7px; }
    .selection-action button { padding: 7px 9px; }
    .empty {
      min-height: 220px;
      display: grid;
      place-items: center;
      text-align: center;
      background: #fff;
      border: 1px dashed #cbd4e1;
      border-radius: 8px;
      padding: 22px;
      color: var(--muted);
    }
    .empty strong { display: block; color: var(--ink); font-size: 16px; margin-bottom: 5px; }
    .status { min-height: 20px; color: #475467; font-size: 13px; margin-top: 10px; }
    .service-status {
      display: flex;
      align-items: center;
      gap: 7px;
      color: #475467;
      font-size: 12px;
      line-height: 1.4;
    }
    .status-dot { width: 7px; height: 7px; flex: 0 0 auto; border-radius: 50%; background: #98a2b3; }
    .status-dot.ready { background: #2f8f57; }
    .status-dot.warn { background: #c07b16; }
    .legend { display: flex; flex-wrap: wrap; gap: 5px; }
    .legend .tag { font-size: 11px; padding: 2px 6px; }
    .sidebar-details { border-top: 1px solid #d5dbe3; margin-top: 16px; padding-top: 11px; }
    .sidebar-details > summary { cursor: pointer; color: #344054; font-size: 12px; font-weight: 700; }
    .sidebar-details[open] > summary { margin-bottom: 10px; }
    .workspace-path {
      margin-top: 8px;
      color: #667085;
      font: 10px/1.4 "Cascadia Mono", Consolas, monospace;
      overflow-wrap: anywhere;
    }
    .build-label { display: none; }
    .section-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .section-title h2 { margin-right: auto; }
    .icon-button { width: 28px; height: 28px; padding: 0; display: grid; place-items: center; font-size: 18px; line-height: 1; }
    .context-row { border-top: 1px solid #d8dee8; padding: 8px 0; }
    .context-row:first-child { border-top: 0; padding-top: 0; }
    .context-head { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; }
    .context-model { color: var(--muted); overflow-wrap: anywhere; }
    .context-meter { height: 5px; margin: 6px 0; background: #dfe5ee; border-radius: 3px; overflow: hidden; }
    .context-meter > span { display: block; height: 100%; background: #1772d0; }
    .context-meter > span.warn { background: #c6372f; }
    .context-detail { color: #526178; font-size: 11px; line-height: 1.35; }
    @media (max-width: 1050px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: relative; height: auto; }
      .layout { grid-template-columns: 1fr; }
      .side { position: relative; top: 0; }
      .reader { height: 70vh; }
      .chat-panel { height: 70vh; min-height: 480px; }
      .judge-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .app-header { align-items: flex-start; flex-direction: column; }
      .stat-grid { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="brand-mark">P</div>
        <div><div class="brand-name">Deep Paper Reader</div><div class="brand-version">local research workspace</div></div>
      </div>
      <div class="sidebar-label">Open paper</div>
      <div class="upload">
        <input id="fileInput" type="file" accept="application/pdf,.pdf" />
        <button id="uploadBtn" class="primary">Upload and parse</button>
        <div id="uploadStatus" class="status">Choose a PDF, then upload and parse.</div>
      </div>
      <div class="sidebar-label">Research</div>
      <div class="service-status"><span id="tavilyDot" class="status-dot"></span><span id="tavilyText">Checking Tavily...</span></div>
      <div class="sidebar-label">Highlights</div>
      <div class="legend">
        <span class="tag" style="background:var(--math)">math</span>
        <span class="tag" style="background:var(--method)">method</span>
        <span class="tag" style="background:var(--result)">result</span>
        <span class="tag" style="background:var(--concept)">concept</span>
      </div>
      <details class="sidebar-details">
        <summary>Runtime details</summary>
        <div class="section-title"><h2>Context usage</h2><button id="refreshContext" class="icon-button" title="Refresh context diagnostics" aria-label="Refresh context diagnostics">&#8635;</button></div>
        <div id="contextDebug" class="subtle">Reading model settings...</div>
        <div id="workspaceText" class="workspace-path">No workspace loaded</div>
      </details>
    </aside>
    <main>
      <header class="app-header">
        <div class="app-heading">
          <h1 id="paperTitle">Paper workspace</h1>
          <div id="paperMeta" class="paper-meta">Open a PDF to begin.</div>
          <span id="buildLabel" class="build-label">reader-build-2026-07-28-aligned-judge-v32</span>
        </div>
        <div id="stats" class="stat-grid"></div>
      </header>
      <div class="layout">
        <section id="reader" class="reader"><div class="empty"><div><strong>No paper open</strong>Choose a PDF from the sidebar to create a local research workspace.</div></div></section>
        <section class="side">
          <div class="side-tabs" role="tablist" aria-label="Paper tools">
            <button id="chatTab" class="side-tab active" role="tab" aria-selected="true" data-side-panel="chat">Chat</button>
            <button id="inspectTab" class="side-tab" role="tab" aria-selected="false" data-side-panel="inspect">Inspect</button>
          </div>
          <div id="chatPanel" class="side-panel chat-panel" role="tabpanel">
            <div class="chat-header">
              <h2>Paper Chat</h2>
              <div id="chatModel" class="chat-model"></div>
              <button id="newChatBtn" class="chat-icon-button" title="Start a new chat" aria-label="Start a new chat">+</button>
            </div>
            <div id="chatMessages" class="chat-messages"><div class="chat-empty">Ask a question about the loaded paper.</div></div>
            <div class="chat-composer">
              <div id="chatModes" class="mode-control" role="group" aria-label="Chat depth">
                <button class="active" data-chat-mode="fast" title="One routed specialist">Fast</button>
                <button data-chat-mode="deep" title="Multiple relevant specialists">Deep</button>
                <button data-chat-mode="web" title="Paper evidence with lazy Tavily research">Web</button>
                <button data-chat-mode="explore" title="Four-perspective paper analysis">Explore</button>
              </div>
              <div class="composer-row">
                <textarea id="chatInput" maxlength="4000" placeholder="Ask about this paper..." aria-label="Paper chat question"></textarea>
                <button id="sendChatBtn" class="send-button" title="Send question" aria-label="Send question">&#8593;</button>
              </div>
              <div id="chatStatus" class="chat-status"></div>
            </div>
          </div>
          <div id="inspectPanel" class="side-panel inspect-panel" role="tabpanel" hidden>
            <div class="card">
              <h2>Sections</h2>
              <div id="sectionList" class="section-list"></div>
            </div>
            <div class="card">
              <h2>Teach the highlighter</h2>
              <div class="memory-form">
                <input id="manualTerm" type="text" maxlength="120" placeholder="Select text or enter a missed term" />
                <select id="manualCategory" aria-label="Highlight category">
                  <option value="concept">Concept</option>
                  <option value="method">Method</option>
                  <option value="math">Math</option>
                  <option value="result">Result</option>
                </select>
                <button id="rememberBtn">Remember highlight</button>
                <div id="memoryStatus" class="subtle">Corrections persist across papers.</div>
              </div>
            </div>
            <div class="card">
              <h2>Significant Terms</h2>
              <div id="termList" class="term-list"></div>
            </div>
            <details class="card">
              <summary><b>Active skills</b></summary>
              <div id="skillList" class="subtle" style="margin-top:10px"></div>
            </details>
          </div>
        </section>
      </div>
    </main>
  </div>
  <div id="popover" class="popover" role="dialog" aria-live="polite"></div>
  <div id="selectionAction" class="selection-action">
    <div id="selectionLabel" class="selection-label"></div>
    <select id="selectionCategory" aria-label="Selected term category">
      <option value="concept">Concept</option>
      <option value="method">Method</option>
      <option value="math">Math</option>
      <option value="result">Result</option>
    </select>
    <button id="selectionSave">Remember</button>
  </div>
  <div id="loadingOverlay" class="loading-overlay" role="status" aria-live="assertive">
    <div class="loading-panel">
      <div class="spinner"></div>
      <h3 id="loadingTitle">Parsing paper</h3>
      <div id="loadingDetail" class="loading-detail">Preparing upload...</div>
      <div class="progress-track"><div id="loadingBar" class="progress-bar"></div></div>
    </div>
  </div>
  <script>
    const BUILD_LABEL = 'reader-build-2026-07-28-aligned-judge-v32';
    const state = { workspace: null, pages: [], sections: [], terms: [], termMap: new Map(), metadata: null, pageLayouts: new Map(), selectionPage: null, mathExplanations: new Map(), mathPending: new Set(), threadId: null, chatMode: 'fast', chatBusy: false, lastQuestion: '', chatQuestions: new Map() };
    const autoWebAttempted = new Set();
    let mathHoverTimer = null;
    const $ = (id) => document.getElementById(id);

    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function newThreadId() {
      return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function threadStorageKey() {
      return `deep-paper-chat:${state.workspace || 'unloaded'}`;
    }

    function setSidePanel(name) {
      const chat = name === 'chat';
      $('chatPanel').hidden = !chat;
      $('inspectPanel').hidden = chat;
      $('chatTab').classList.toggle('active', chat);
      $('inspectTab').classList.toggle('active', !chat);
      $('chatTab').setAttribute('aria-selected', String(chat));
      $('inspectTab').setAttribute('aria-selected', String(!chat));
    }

    function formatChatInline(text) {
      return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
    }

    function formatChatMarkdown(text) {
      const chunks = String(text || '').split(/```/);
      return chunks.map((chunk, index) => {
        if (index % 2 === 1) {
          const body = chunk.replace(/^[A-Za-z0-9_+-]+\n/, '');
          return `<pre><code>${escapeHtml(body.trim())}</code></pre>`;
        }
        return chunk.split(/\n{2,}/).map(block => {
          const clean = block.trim();
          if (!clean) return '';
          const lines = clean.split('\n');
          if (lines.every(line => /^\s*[-*]\s+/.test(line))) {
            return `<ul>${lines.map(line => `<li>${formatChatInline(line.replace(/^\s*[-*]\s+/, ''))}</li>`).join('')}</ul>`;
          }
          if (lines.every(line => /^\s*\d+[.)]\s+/.test(line))) {
            return `<ol>${lines.map(line => `<li>${formatChatInline(line.replace(/^\s*\d+[.)]\s+/, ''))}</li>`).join('')}</ol>`;
          }
          const heading = clean.match(/^#{1,3}\s+(.+)$/);
          if (heading) return `<h4>${formatChatInline(heading[1])}</h4>`;
          return `<p>${lines.map(formatChatInline).join('<br>')}</p>`;
        }).join('');
      }).join('');
    }

    function citationMarkup(citations) {
      return (citations || []).map(item => {
        const attrs = item.page
          ? `data-chat-page="${Number(item.page)}"`
          : (item.url ? `data-chat-url="${escapeHtml(item.url)}"` : '');
        return `<button class="citation-button ${item.source === 'web' ? 'web' : ''}" ${attrs} title="${escapeHtml(item.excerpt || item.title || '')}">${escapeHtml(item.label || item.id)}</button>`;
      }).join('');
    }

    function traceMarkup(trace, route, verification) {
      const items = (trace || []).map(item => `
        <div class="trace-item"><b>${escapeHtml(item.stage || '')}</b><span>${escapeHtml(item.message || '')} · ${escapeHtml(item.elapsedSeconds ?? 0)}s</span></div>
      `).join('');
      const specialists = route?.specialists || [];
      const score = verification?.score ?? 0;
      return `
        <details class="chat-trace">
          <summary>${escapeHtml(specialists.join(' + ') || 'general')} · grounding ${escapeHtml(Math.round(Number(score) * 100))}%</summary>
          <div class="trace-list">${items || '<div>No trace recorded.</div>'}</div>
        </details>`;
    }

    function assistantMarkup(message, question='') {
      const citations = message.citations || [];
      const verification = message.verification || {};
      const route = message.route || {};
      const messageId = message.id || message.message_id || message.messageId || newThreadId();
      state.chatQuestions.set(messageId, question || state.lastQuestion || '');
      return `
        <div class="chat-message assistant" data-message-id="${escapeHtml(messageId)}">
          <div class="chat-bubble">
            <div class="chat-answer">${formatChatMarkdown(message.content || message.answer || '')}</div>
            ${citations.length ? `<div class="chat-citations">${citationMarkup(citations)}</div>` : ''}
            <div class="chat-meta">
              ${(route.specialists || []).map(name => `<span class="chat-route">${escapeHtml(name)}</span>`).join('')}
              <span>${verification.passed ? 'grounded' : 'reviewed'}</span>
            </div>
            ${traceMarkup(message.trace, route, verification)}
            <div class="chat-feedback">
              <button data-chat-rating="helpful" title="Helpful" aria-label="Helpful">+</button>
              <button data-chat-rating="not_helpful" title="Needs correction" aria-label="Needs correction">-</button>
            </div>
            <div class="chat-correction">
              <textarea maxlength="4000" placeholder="What should the answer have said?" aria-label="Answer correction"></textarea>
              <label class="memory-check"><input type="checkbox" /> Use this correction in future paper chats</label>
              <button data-save-correction>Save correction</button>
              <div class="subtle" data-feedback-status></div>
            </div>
          </div>
        </div>`;
    }

    function renderChatHistory(messages) {
      state.chatQuestions = new Map();
      let lastQuestion = '';
      const html = (messages || []).map(message => {
        if (message.role === 'user') {
          lastQuestion = message.content || '';
          return `<div class="chat-message user"><div class="chat-bubble">${escapeHtml(lastQuestion)}</div></div>`;
        }
        return assistantMarkup(message, lastQuestion);
      }).join('');
      $('chatMessages').innerHTML = html || '<div class="chat-empty">Ask a question about the loaded paper.</div>';
      $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
      if (globalThis.MathJax?.typesetPromise) MathJax.typesetPromise([$('chatMessages')]).catch(() => {});
    }

    async function loadChatHistory() {
      if (!state.workspace || !state.threadId) return;
      const res = await fetch(`/api/chat-history?workspace=${encodeURIComponent(state.workspace)}&thread=${encodeURIComponent(state.threadId)}`);
      if (!res.ok) throw new Error(await res.text());
      const thread = await res.json();
      renderChatHistory(thread.messages || []);
    }

    function setChatBusy(busy, message='') {
      state.chatBusy = busy;
      $('sendChatBtn').disabled = busy || !state.workspace;
      $('chatInput').disabled = busy;
      $('chatStatus').textContent = message;
    }

    function appendChatThinking(message) {
      document.querySelector('[data-chat-thinking]')?.remove();
      $('chatMessages').insertAdjacentHTML('beforeend', `
        <div class="chat-message assistant" data-chat-thinking>
          <div class="chat-thinking"><div class="spinner"></div><span>${escapeHtml(message)}</span></div>
        </div>`);
      $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
    }

    async function pollChatJob(jobId) {
      const started = Date.now();
      while (Date.now() - started < 1200000) {
        await new Promise(resolve => setTimeout(resolve, 700));
        const res = await fetch(`/api/chat-status?job=${encodeURIComponent(jobId)}`);
        if (!res.ok) throw new Error(await res.text());
        const job = await res.json();
        const status = `${job.message || 'Working...'} (${job.elapsedSeconds || 0}s)`;
        setChatBusy(true, status);
        appendChatThinking(status);
        if (job.status === 'completed') return job.result;
        if (job.status === 'failed') throw new Error(job.message || 'Paper chat failed.');
      }
      throw new Error('Paper chat exceeded the client timeout.');
    }

    async function sendChat() {
      const question = $('chatInput').value.trim();
      if (!question || !state.workspace || state.chatBusy) return;
      state.lastQuestion = question;
      const empty = $('chatMessages').querySelector('.chat-empty');
      if (empty) empty.remove();
      $('chatMessages').insertAdjacentHTML('beforeend', `<div class="chat-message user"><div class="chat-bubble">${escapeHtml(question)}</div></div>`);
      $('chatInput').value = '';
      appendChatThinking('Planning paper evidence...');
      setChatBusy(true, 'Planning paper evidence...');
      try {
        const res = await fetch(`/api/chat?workspace=${encodeURIComponent(state.workspace)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question, mode: state.chatMode, threadId: state.threadId}),
        });
        if (!res.ok) throw new Error(await res.text());
        const queued = await res.json();
        state.threadId = queued.threadId || state.threadId;
        localStorage.setItem(threadStorageKey(), state.threadId);
        const result = await pollChatJob(queued.jobId);
        document.querySelector('[data-chat-thinking]')?.remove();
        $('chatMessages').insertAdjacentHTML('beforeend', assistantMarkup(result, question));
        $('chatMessages').scrollTop = $('chatMessages').scrollHeight;
        if (globalThis.MathJax?.typesetPromise) await MathJax.typesetPromise([$('chatMessages')]);
        setChatBusy(false, `${(result.context?.evidenceItems || 0)} evidence items · ${(result.context?.specialists || []).join(' + ')}`);
        loadContextDiagnostics().catch(() => {});
      } catch (err) {
        document.querySelector('[data-chat-thinking]')?.remove();
        $('chatMessages').insertAdjacentHTML('beforeend', `<div class="chat-message assistant"><div class="chat-bubble"><div class="chat-answer"><p>${escapeHtml(err.message)}</p></div></div></div>`);
        setChatBusy(false, 'Chat stopped.');
      }
    }

    async function submitChatFeedback(messageElement, rating, correction='', remember=false) {
      const messageId = messageElement.dataset.messageId || '';
      const res = await fetch(`/api/chat-feedback?workspace=${encodeURIComponent(state.workspace)}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          threadId: state.threadId,
          messageId,
          rating,
          question: state.chatQuestions.get(messageId) || '',
          correction,
          remember,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function submitMathJudgeFeedback(rating, dimension='overall', feedback='', remember=true) {
      const pop = $('popover');
      const explanation = state.mathExplanations.get(pop.dataset.mathKey);
      if (!explanation) throw new Error('The equation explanation is no longer available.');
      const compactExplanation = {
        equation_id: explanation.equation_id,
        page: explanation.page,
        display_latex: explanation.display_latex,
        role: explanation.role,
        plain_english: explanation.plain_english,
        context_fit: explanation.context_fit,
        symbols: explanation.symbols,
        evaluation: explanation.evaluation,
      };
      const res = await fetch(`/api/math-judge-feedback?workspace=${encodeURIComponent(state.workspace)}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({rating, dimension, feedback, remember, explanation: compactExplanation}),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function renderPdfPage(page) {
      const ratio = `${Number(page.width) || 612} / ${Number(page.height) || 792}`;
      const workspace = encodeURIComponent(state.workspace);
      return `
        <article id="pdf-page-${page.page}" class="pdf-page-shell" data-page="${page.page}">
          <div class="pdf-page-label">Page ${page.page}</div>
          <div class="pdf-page" style="aspect-ratio:${ratio}">
            <img loading="lazy" alt="Original PDF page ${page.page}" src="/api/page-image?workspace=${workspace}&page=${page.page}" />
            <div class="text-layer" aria-label="Selectable text for page ${page.page}"></div>
            <div class="highlight-layer"></div>
            <div class="math-layer"></div>
          </div>
        </article>`;
    }

    function percent(value, total) {
      return `${(Number(value) / Number(total) * 100).toFixed(5)}%`;
    }

    async function loadPageLayout(pageNumber) {
      if (state.pageLayouts.has(pageNumber)) return;
      state.pageLayouts.set(pageNumber, 'loading');
      try {
        const res = await fetch(`/api/page-layout?workspace=${encodeURIComponent(state.workspace)}&page=${pageNumber}`);
        if (!res.ok) throw new Error(await res.text());
        const layout = await res.json();
        state.pageLayouts.set(pageNumber, layout);
        const pageShell = document.getElementById(`pdf-page-${pageNumber}`);
        if (!pageShell) return;
        const textLayer = pageShell.querySelector('.text-layer');
        const highlightLayer = pageShell.querySelector('.highlight-layer');
        const mathLayer = pageShell.querySelector('.math-layer');
        textLayer.innerHTML = layout.words.map(word => `
          <span class="text-word" style="left:${percent(word.x, layout.width)};top:${percent(word.y, layout.height)};width:${percent(word.width, layout.width)};height:${percent(word.height, layout.height)};font-size:${(Number(word.height) / Number(layout.width) * 92).toFixed(4)}cqw">${escapeHtml(word.text)} </span>
        `).join('');
        highlightLayer.innerHTML = layout.highlights.map(mark => `
          <button class="pdf-highlight ${escapeHtml(mark.category)} ${mark.learned ? 'learned' : ''}"
            style="left:${percent(mark.x, layout.width)};top:${percent(mark.y, layout.height)};width:${percent(mark.width, layout.width)};height:${percent(mark.height, layout.height)}"
            data-term="${escapeHtml(mark.term)}" title="${escapeHtml(mark.term)}"></button>
        `).join('');
        mathLayer.innerHTML = (layout.mathRegions || []).flatMap(region =>
          (region.rects?.length ? region.rects : [region]).map(rect => `
            <button class="math-hotspot ${escapeHtml(region.kind || 'display')}"
              style="left:${percent(rect.x, layout.width)};top:${percent(rect.y, layout.height)};width:${percent(rect.width, layout.width)};height:${percent(rect.height, layout.height)}"
              data-math-page="${layout.page}" data-math-region="${escapeHtml(region.id)}"
              title="Hover to explain this ${region.kind === 'inline' ? 'notation' : 'equation'}"></button>
          `)
        ).join('');
      } catch (err) {
        state.pageLayouts.delete(pageNumber);
        console.error(`Page ${pageNumber} overlay failed`, err);
      }
    }

    function observePages() {
      const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          loadPageLayout(Number(entry.target.dataset.page));
          observer.unobserve(entry.target);
        }
      }, {root: $('reader'), rootMargin: '900px 0px'});
      document.querySelectorAll('.pdf-page-shell').forEach(page => observer.observe(page));
    }

    function renderSectionList() {
      $('sectionList').innerHTML = state.sections.length
        ? state.sections.map(section => `<button class="section-link" data-page-target="${section.page}">${escapeHtml(section.label)}</button>`).join('')
        : state.pages.map(page => `<button class="section-link" data-page-target="${page.page}">Page ${page.page}</button>`).join('');
    }

    function renderState(payload) {
      state.workspace = payload.workspace;
      state.pages = payload.pages || [];
      state.sections = payload.sections || [];
      state.terms = payload.terms || [];
      state.metadata = payload.metadata || {};
      state.pageLayouts = new Map();
      state.mathExplanations = new Map();
      state.mathPending = new Set();
      state.termMap = new Map(state.terms.map(t => [String(t.term).toLowerCase(), t]));
      state.threadId = localStorage.getItem(threadStorageKey()) || newThreadId();
      localStorage.setItem(threadStorageKey(), state.threadId);
      $('buildLabel').textContent = payload.build || BUILD_LABEL;
      $('workspaceText').textContent = payload.workspace || 'No workspace loaded';
      const title = String(state.metadata.title_guess || '').trim();
      $('paperTitle').textContent = title || 'Paper workspace';
      $('paperTitle').title = title;
      $('paperMeta').textContent = `${state.metadata.page_count || state.pages.length || 0} pages · local workspace`;
      $('tavilyText').textContent = payload.tavilyReady
        ? 'Tavily ready on term open'
        : 'Tavily key not configured';
      $('tavilyDot').className = `status-dot ${payload.tavilyReady ? 'ready' : 'warn'}`;
      $('stats').innerHTML = [
        ['Pages', state.metadata.page_count || state.pages.length || 0],
        ['Terms', state.terms.length],
        ['Equations', payload.equationCount || 0],
        ['Figures/Tables', `${payload.figureCount || 0}/${payload.tableCount || 0}`],
      ].map(([label, value]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join('');
      $('reader').innerHTML = state.pages.length ? state.pages.map(renderPdfPage).join('') : '<div class="empty">No PDF pages found.</div>';
      renderSectionList();
      renderTerms();
      renderSkills(payload.skills || []);
      $('chatModel').textContent = payload.chat?.model || '';
      setChatBusy(false, '');
      loadChatHistory().catch(err => {
        $('chatStatus').textContent = `Chat history unavailable: ${err.message}`;
      });
      observePages();
      loadContextDiagnostics().catch(err => {
        $('contextDebug').textContent = `Context diagnostics unavailable: ${err.message}`;
      });
    }

    function renderTerms() {
      $('termList').innerHTML = state.terms.map(term =>
        `<button class="pill ${escapeHtml(term.category)}" data-term="${escapeHtml(term.term)}" title="${term.learned ? 'Learned from a user correction' : 'Detected by the paper-term skill'}">${escapeHtml(term.term)} · ${escapeHtml(term.score)}${term.learned ? ' · learned' : ''}</button>`
      ).join('');
    }

    function renderSkills(skills) {
      $('skillList').innerHTML = skills.map(s => `<div><b>${escapeHtml(s.name)}</b><br>${escapeHtml(s.description || '').slice(0, 130)}</div>`).join('<hr>');
    }

    function compactTokens(value) {
      const count = Number(value);
      if (!Number.isFinite(count) || count <= 0) return 'unknown';
      if (count >= 1000000) return `${(count / 1000000).toFixed(count >= 10000000 ? 0 : 1)}m`;
      if (count >= 1000) return `${(count / 1000).toFixed(count >= 10000 ? 0 : 1)}k`;
      return String(Math.round(count));
    }

    function renderContextDiagnostics(payload) {
      const models = payload.models || [];
      const runtime = payload.ollamaRuntime || {};
      const hardware = payload.hardwareProfile || null;
      const loaded = runtime.models || [];
      const hardwareSummary = hardware
        ? `<div class="context-detail"><b>${escapeHtml(hardware.name)} profile</b> · ${compactTokens(hardware.contextLength)} context · ${escapeHtml(hardware.gpu?.name || 'CPU')}</div>`
        : '<div class="context-detail"><b>Hardware profile not saved.</b><br>Run <code>paper-agent hardware --apply</code>.</div>';
      const runtimeSummary = runtime.available
        ? `<div class="context-detail"><b>${loaded.length} loaded</b> · ${(Number(runtime.totalVramBytes || 0) / 1073741824).toFixed(1)} GB VRAM<br>${loaded.map(item => `${escapeHtml(item.model)} (${compactTokens(item.contextLength)})`).join(' · ') || 'No model currently resident.'}</div>`
        : '<div class="context-detail">Ollama runtime status unavailable.</div>';
      const modelRows = models.length ? models.map(item => {
        const latest = item.latest || null;
        const percent = Number(item.estimatedEffectiveUsagePercent || 0);
        const width = Math.max(0, Math.min(100, percent));
        const configured = compactTokens(item.configuredWindowTokens);
        const nativeWindow = compactTokens(item.nativeWindowTokens);
        const latestDetail = latest
          ? `${compactTokens(latest.estimatedInputTokens)} estimated prompt + ${compactTokens(latest.reservedOutputTokens)} reserved = ${percent.toFixed(1)}%`
          : 'No prompt recorded for this workflow yet.';
        return `
          <div class="context-row">
            <div class="context-head"><b>${escapeHtml(item.label)}</b><span>${escapeHtml(configured)} active</span></div>
            <div class="context-model">${escapeHtml(item.model)} · native ${escapeHtml(nativeWindow)}${item.latestModel && item.latestModel !== item.model ? `<br>latest used: ${escapeHtml(item.latestModel)}` : ''}</div>
            <div class="context-meter"><span class="${percent >= 85 ? 'warn' : ''}" style="width:${width}%"></span></div>
            <div class="context-detail">${escapeHtml(latestDetail)}</div>
          </div>`;
      }).join('') : '<div class="context-detail">No model workflows configured.</div>';
      $('contextDebug').innerHTML = `${hardwareSummary}${runtimeSummary}${modelRows}`;
    }

    async function loadContextDiagnostics() {
      const query = state.workspace ? `?workspace=${encodeURIComponent(state.workspace)}` : '';
      const res = await fetch(`/api/context-diagnostics${query}`);
      if (!res.ok) throw new Error(await res.text());
      renderContextDiagnostics(await res.json());
    }

    function setLoading(open, detail='') {
      const overlay = $('loadingOverlay');
      const uploadBox = document.querySelector('.upload');
      overlay.classList.toggle('open', open);
      $('reader').classList.toggle('busy', open);
      uploadBox?.classList.toggle('busy', open);
      if (detail) {
        $('loadingDetail').textContent = detail;
        $('uploadStatus').textContent = detail;
      }
    }

    function setProgress(percent, detail) {
      $('loadingBar').style.width = `${Math.max(8, Math.min(100, percent))}%`;
      setLoading(true, detail);
    }

    function clearLoading(message) {
      setLoading(false);
      $('loadingBar').style.width = '18%';
      if (message) $('uploadStatus').textContent = message;
    }

    function nextPaint() {
      return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }

    async function loadState(workspace=null) {
      const url = workspace ? `/api/state?workspace=${encodeURIComponent(workspace)}` : '/api/state';
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      renderState(await res.json());
    }

    async function uploadPdf() {
      const file = $('fileInput').files[0];
      if (!file) {
        $('uploadStatus').textContent = 'Choose a PDF first.';
        return;
      }
      $('uploadBtn').disabled = true;
      $('fileInput').disabled = true;
      $('reader').innerHTML = `
        <div class="empty">
          <div><strong>Parsing ${escapeHtml(file.name)}</strong>Extracting paper structure and research terms.</div>
        </div>
      `;
      setProgress(12, `Preparing ${file.name}...`);
      try {
        await nextPaint();
        setProgress(28, 'Uploading PDF to local parser...');
        await nextPaint();
        const uploadPromise = fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/pdf'},
          body: file,
        });
        const stagedMessages = [
          [45, 'Extracting page text and metadata...'],
          [62, 'Detecting equations, figures, and tables...'],
          [78, 'Scoring significant method, math, and result terms...'],
          [88, 'Building highlighted reader view...'],
        ];
        let stageIndex = 0;
        const stageTimer = setInterval(() => {
          if (stageIndex < stagedMessages.length) {
            const [percent, message] = stagedMessages[stageIndex];
            setProgress(percent, message);
            stageIndex += 1;
          }
        }, 1200);
        const res = await uploadPromise;
        clearInterval(stageTimer);
        if (!res.ok) throw new Error(await res.text());
        setProgress(96, 'Rendering paper...');
        const payload = await res.json();
        renderState(payload);
        clearLoading('Parsed and highlighted.');
      } catch (err) {
        clearLoading(`Parse failed: ${err.message}`);
        $('reader').innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      } finally {
        $('uploadBtn').disabled = false;
        $('fileInput').disabled = false;
      }
    }

    async function openTerm(term) {
      if (!term || !state.workspace) return;
      const pop = $('popover');
      const autoKey = `${state.workspace}::${term.toLowerCase()}`;
      delete pop.dataset.mathKey;
      pop.dataset.termKey = autoKey;
      pop.classList.add('open');
      pop.innerHTML = `<div class="popover-header"><h3>${escapeHtml(term)}</h3><button id="closePop">Close</button></div><div class="subtle">Loading term card...</div>`;
      $('closePop').onclick = () => pop.classList.remove('open');
      const res = await fetch(`/api/term?workspace=${encodeURIComponent(state.workspace)}&term=${encodeURIComponent(term)}`);
      if (pop.dataset.termKey !== autoKey) return;
      if (!res.ok) {
        pop.innerHTML = `<div class="popover-header"><h3>${escapeHtml(term)}</h3><button id="closePop">Close</button></div><p>${escapeHtml(await res.text())}</p>`;
        $('closePop').onclick = () => pop.classList.remove('open');
        return;
      }
      const data = await res.json();
      if (pop.dataset.termKey !== autoKey) return;
      const card = data.card || {};
      const sig = data.significantTerm;
      const equations = data.equations || [];
      const visuals = data.visuals || {figures: [], tables: []};
      const skillTrace = data.skillTrace || [];
      const webSources = (card.web_sources || []);
      const generalReady = ['tavily_research', 'tavily_search_answer'].includes(card.general_explanation_source);
      const whyReady = card.why_it_matters_source === 'paper_agent';
      const needsEnrichment = !generalReady || !whyReady;
      const generalLabel = card.general_explanation_source === 'tavily_research'
        ? `Tavily Research · ${card.general_explanation_model || 'auto'}`
        : (card.general_explanation_source === 'tavily_search_answer'
            ? 'Tavily Search fallback'
            : (data.tavilyReady ? 'Researching' : 'Not enriched'));
      const whyLabel = whyReady
        ? `Paper agent · ${card.why_it_matters_model || 'configured model'}`
        : (card.why_it_matters_source === 'paper_agent_error'
            ? 'Paper agent failed'
            : (data.tavilyReady ? 'Pending analysis' : 'Not synthesized'));
      const generalContent = generalReady
        ? `<div class="concept-copy">${formatConceptText(card.general_explanation || '')}</div>`
        : (data.tavilyReady
            ? '<div class="enrichment-placeholder">Researching a sourced general explanation...</div>'
            : `<div class="concept-copy">${formatConceptText(card.general_explanation || 'Tavily is not configured.')}</div>`);
      const whyContent = whyReady
        ? `<div class="concept-copy">${formatConceptText(card.why_it_matters_here || '')}</div>`
        : (data.tavilyReady
            ? '<div class="enrichment-placeholder">Waiting for the paper-grounded relevance analysis...</div>'
            : `<div class="concept-copy">${formatConceptText(card.why_it_matters_here || '')}</div>`);
      const evidence = (card.paper_evidence || []).map(cleanConceptEvidence).filter(Boolean);
      const hasVisuals = Boolean(visuals.figures?.length || visuals.tables?.length);
      pop.innerHTML = `
        <div class="popover-header">
          <div>
            <h3>${escapeHtml(term)}</h3>
            <div class="tag-row">${skillTrace.map(s => `<span class="tag">${escapeHtml(s.name)}</span>`).join('')}</div>
          </div>
          <button id="closePop">Close</button>
        </div>
        ${sig ? `<div class="concept-meta"><span>Category <b>${escapeHtml(sig.category)}</b></span><span>Score ${escapeHtml(sig.score)}</span><span>${escapeHtml(sig.paper_occurrences)} paper mentions</span><span>${escapeHtml(sig.method_occurrences)} method mentions</span></div>` : ''}
        <div class="concept-status">
          <span id="enrichStatus" class="enrich-status">${
            generalReady && whyReady
              ? `Enriched · ${webSources.length} research source(s)`
              : (data.tavilyReady ? 'Running Tavily Research, then the paper agent...' : 'Tavily not configured.')
          }</span>
        </div>
        <section class="concept-section">
          <div class="concept-section-heading"><h4>In this paper</h4></div>
          <div class="concept-copy">${formatConceptText(card.paper_specific_meaning || 'No paper-specific card yet.')}</div>
          ${(card.prerequisites || []).length ? `<div class="tag-row">${card.prerequisites.map(item => `<span class="tag">${escapeHtml(item)}</span>`).join('')}</div>` : ''}
        </section>
        <section class="concept-section">
          <div class="concept-section-heading"><h4>General explanation</h4><span class="provenance">${escapeHtml(generalLabel)}</span></div>
          ${generalContent}
        </section>
        <section class="concept-section">
          <div class="concept-section-heading"><h4>Why it matters here</h4><span class="provenance">${escapeHtml(whyLabel)}</span></div>
          ${whyContent}
        </section>
        <section class="concept-section">
          <div class="concept-section-heading"><h4>Paper evidence</h4></div>
          <ul class="evidence">${evidence.map(item => `<li>${formatTechnicalInline(item)}</li>`).join('') || '<li>No direct sentence evidence found.</li>'}</ul>
        </section>
        <details class="concept-details" open>
          <summary>Research lineage</summary>
          <div id="lineageContent" class="subtle">Tracing this term through the paper's citations...</div>
        </details>
        ${equations.length ? `<details class="concept-details"><summary>Related equations (${equations.length})</summary>${equations.map(eq => `<div class="equation-card">${escapeHtml(eq.id)} · page ${escapeHtml(eq.page)}\n${escapeHtml(eq.raw)}</div>`).join('')}</details>` : ''}
        ${hasVisuals ? `<details class="concept-details"><summary>Related figures and tables (${(visuals.figures || []).length + (visuals.tables || []).length})</summary>${(visuals.figures || []).map(f => `<p><b>${escapeHtml(f.id)}</b> · page ${escapeHtml(f.page)}<br>${escapeHtml(f.caption)}</p>`).join('')}${(visuals.tables || []).map(t => `<p><b>${escapeHtml(t.id)}</b> · page ${escapeHtml(t.page)}<br>${escapeHtml(t.caption)}</p>`).join('')}</details>` : ''}
        <details class="concept-details"><summary>Research sources (${webSources.length})</summary><div id="webSources">${renderSources(webSources)}</div></details>
      `;
      $('closePop').onclick = () => pop.classList.remove('open');
      loadLineage(term);
      if (data.tavilyReady && needsEnrichment && !autoWebAttempted.has(autoKey)) {
        autoWebAttempted.add(autoKey);
        enrichTerm(term);
      }
      typesetMath(pop);
    }

    function normalizeInlineLatex(value) {
      return String(value || '')
        .replace(/^\$|\$$/g, '')
        .replace(/^\\\(|\\\)$/g, '')
        .replace(/\u0008/g, '\\b')
        .replace(/\u0009/g, '\\t')
        .replace(/\u000a/g, '\\n')
        .replace(/\u000c/g, '\\f')
        .replace(/\u000d/g, '\\r')
        .replace(/\\{2,}(?=[A-Za-z])/g, '\\')
        .replaceAll('\\nlimits', '\\limits')
        .replace(/\\boldsymbol/g, '\\mathbf')
        .replace(/(^|[^A-Za-z\\])ext\{/g, '$1\\text{')
        .replace(/(^|[^A-Za-z\\])imes\b/g, '$1\\times')
        .replace(/(^|[^A-Za-z\\])oldsymbol\{/g, '$1\\mathbf{')
        .replace(/(^|[^A-Za-z\\])rac\{/g, '$1\\frac{')
        .replace(/(^|[^A-Za-z\\])mathcal\{/g, '$1\\mathcal{')
        .trim();
    }

    function inlineLatexIsSafe(latex) {
      if (!latex || /[\u0000-\u001f]/.test(latex)) return false;
      if ((latex.match(/\{/g) || []).length !== (latex.match(/\}/g) || []).length) return false;
      if (/\\(?:href|url|includegraphics|input|write18)\b/i.test(latex)) return false;
      if (/(^|[^A-Za-z\\])(?:ext|imes|oldsymbol|rac)(?:\{|\b)/.test(latex)) return false;
      return !/\\(?:n|t|b|r|f)(?![A-Za-z])/.test(latex);
    }

    function readableInlineMath(value) {
      return normalizeInlineLatex(value)
        .replace(/\\(?:text|mathrm|mathbf|mathcal|mathbb|operatorname)\{([^{}]*)\}/g, '$1')
        .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
        .replace(/\\(?:leq|le)\b/g, '≤')
        .replace(/\\(?:geq|ge)\b/g, '≥')
        .replace(/\\(?:neq|ne)\b/g, '≠')
        .replace(/\\in\b/g, '∈')
        .replace(/\\times\b/g, '×')
        .replace(/\\cdot\b/g, '·')
        .replace(/\\sum\b/g, 'Σ')
        .replace(/\\[A-Za-z]+/g, '')
        .replace(/[{}]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    }

    function renderInlineMath(token) {
      const latex = normalizeInlineLatex(token);
      if (!inlineLatexIsSafe(latex)) {
        return `<code class="inline-math-fallback">${escapeHtml(readableInlineMath(latex) || latex)}</code>`;
      }
      return `<span class="inline-math-source" data-latex="${escapeHtml(latex)}">\\(${escapeHtml(latex)}\\)</span>`;
    }

    function formatInline(text) {
      const source = String(text || '');
      const tokenPattern = /(`[^`\n]*`|\*\*[^*\n]+\*\*|\$[\s\S]*?\$|\\\([\s\S]*?\\\))/g;
      let output = '';
      let cursor = 0;
      let match;
      while ((match = tokenPattern.exec(source)) !== null) {
        output += escapeHtml(source.slice(cursor, match.index));
        const token = match[0];
        if (token.startsWith('`')) {
          output += `<code>${escapeHtml(token.slice(1, -1))}</code>`;
        } else if (token.startsWith('**')) {
          output += `<strong>${formatInline(token.slice(2, -2))}</strong>`;
        } else {
          output += renderInlineMath(token);
        }
        cursor = tokenPattern.lastIndex;
      }
      return output + escapeHtml(source.slice(cursor));
    }

    function plainMathToLatex(value) {
      return String(value || '')
        .replace(/^\$|\$$/g, '')
        .replace(/^\\\(|\\\)$/g, '')
        .replace(/ℝ/g, '\\mathbb{R}')
        .replace(/∈/g, '\\in ')
        .replace(/×/g, ' \\times ')
        .replace(/√\s*([A-Za-z][A-Za-z0-9_{}]*)/g, '\\sqrt{$1}')
        .replace(/\bsoftmax\b/gi, '\\operatorname{softmax}')
        .replace(/\bbatch\b/g, '\\mathrm{batch}')
        .trim();
    }

    function wrapBracketedLatex(value) {
      const source = String(value || '');
      let output = '';
      let cursor = 0;
      while (cursor < source.length) {
        if (source[cursor] !== '[') {
          output += source[cursor];
          cursor += 1;
          continue;
        }
        let depth = 1;
        let end = cursor + 1;
        while (end < source.length && depth > 0) {
          if (source[end] === '[') depth += 1;
          if (source[end] === ']') depth -= 1;
          end += 1;
        }
        if (depth !== 0) {
          output += source[cursor];
          cursor += 1;
          continue;
        }
        const content = source.slice(cursor + 1, end - 1);
        const hasLatex = /\\(?:mathcal|mathbb|mathbf|mathrm|text|operatorname|frac|sum|bigcup)\b/.test(content);
        const bracesBalanced = (content.match(/\{/g) || []).length === (content.match(/\}/g) || []).length;
        output += hasLatex && bracesBalanced ? `$${content}$` : source.slice(cursor, end);
        cursor = end;
      }
      return output;
    }

    function formatTechnicalInline(text) {
      const source = wrapBracketedLatex(text);
      const math = /(\$[^$\n]+\$|\\\([^\n]*?\\\)|[A-Za-z](?:_[A-Za-z0-9]+)?\s*∈\s*ℝ\^\{[^}]+\}|[A-Za-z](?:_[A-Za-z0-9]+)?\s*=\s*[A-Za-z0-9_^{}]+|softmax\([^)]{1,140}\)\s*[A-Za-z](?:_[A-Za-z0-9]+)?|batch(?:×[A-Za-z0-9_]+){2,})/g;
      let output = '';
      let cursor = 0;
      let match;
      while ((match = math.exec(source)) !== null) {
        output += formatInline(source.slice(cursor, match.index));
        output += renderInlineMath(plainMathToLatex(match[0]));
        cursor = math.lastIndex;
      }
      output += formatInline(source.slice(cursor));
      return output;
    }

    function formatConceptText(text) {
      const source = String(text || '').trim();
      if (!source) return '<p>No explanation is available.</p>';
      let blocks = source.split(/\n{2,}/).map(item => item.trim()).filter(Boolean);
      if (blocks.length === 1) {
        const sentences = source.split(/(?<=[.!?])\s+(?=[A-Z])/).filter(Boolean);
        blocks = [];
        for (let index = 0; index < sentences.length; index += 2) {
          blocks.push(sentences.slice(index, index + 2).join(' '));
        }
      }
      return blocks.map(block => `<p>${formatTechnicalInline(block)}</p>`).join('');
    }

    function cleanConceptEvidence(text) {
      let clean = String(text || '').replace(/[\u0000-\u001f]/g, ' ').replace(/\s+/g, ' ').trim();
      const colon = clean.indexOf(':');
      if (colon > 44) {
        const tail = clean.slice(colon + 1);
        const noise = (tail.match(/[=∈∑∏≤≥]/g) || []).length + (tail.match(/(?:Top-k|Score\w*|\w+Comp\w*)/gi) || []).length;
        if (noise >= 2) clean = `${clean.slice(0, colon).replace(/[ .]+$/, '')}.`;
      }
      return clean;
    }

    function cleanListItem(text) {
      const clean = String(text || '')
        .replace(/^\s*(?:step\s*)?\d+\s*[.)-]\s*/i, '')
        .replace(/^\s*[-*]\s*/, '')
        .trim();
      if (/tensor_dims_explanation|implementation_view_intuition|symbol_meanings_table|equation_function_classification/i.test(clean)) return '';
      if ((clean.match(/_/g) || []).length >= 4 && /^[a-z0-9_]+$/i.test(clean)) return '';
      return clean;
    }

    function cleanPaperEvidence(text) {
      let clean = String(text || '')
        .replace(/[\u0000-\u001f\uE000-\uF8FF]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/^\(\d+\)\s*[•·]?\s*/, '');
      const step = clean.search(/(?:•\s*)?Step\s+\d+\s*:/i);
      if (step > 0) clean = clean.slice(step).replace(/^•\s*/, '');
      const colon = clean.lastIndexOf(':');
      if (colon > 60) {
        const tail = clean.slice(colon + 1);
        const noise = (tail.match(/[=∈∑]/g) || []).length;
        if (noise >= 2) clean = `${clean.slice(0, colon).trim()}.`;
      }
      if (clean.length > 520) clean = `${clean.slice(0, 520).replace(/\s+\S*$/, '').replace(/[ ,;:]+$/, '')}...`;
      return clean;
    }

    function humanizeLabel(value, fallback='Not classified') {
      const clean = String(value || fallback).replace(/_+/g, ' ').replace(/\s+/g, ' ').trim();
      return clean ? clean.charAt(0).toUpperCase() + clean.slice(1) : fallback;
    }

    function renderImplementation(text) {
      const source = String(text || '').trim();
      if (!source) return '<p class="subtle">Implementation view was not resolved.</p>';
      const fence = /```(?:[A-Za-z0-9_+-]+)?\s*\n?([\s\S]*?)```/g;
      let output = '';
      let cursor = 0;
      let match;
      while ((match = fence.exec(source)) !== null) {
        const prose = source.slice(cursor, match.index).trim();
        if (prose) output += `<p>${formatInline(prose)}</p>`;
        output += `<pre class="implementation-code"><code>${escapeHtml(match[1].trim())}</code></pre>`;
        cursor = fence.lastIndex;
      }
      const tail = source.slice(cursor).trim();
      if (tail) output += `<p>${formatInline(tail)}</p>`;
      if (!output) {
        const looksLikeCode = /(^|\n)\s*(?:import |from |def |class |return |[A-Za-z_]\w*\s*=)/m.test(source);
        return looksLikeCode
          ? `<pre class="implementation-code"><code>${escapeHtml(source)}</code></pre>`
          : `<p>${formatInline(source)}</p>`;
      }
      return output;
    }

    function renderSources(sources) {
      if (!sources?.length) return '<p class="subtle">No cached web sources for this term.</p>';
      return sources.map(src => `
        <div class="source-card">
          <a href="${escapeHtml(src.url || '#')}" target="_blank" rel="noreferrer">${escapeHtml(src.title || src.url || 'Untitled source')}</a>
          <div class="subtle">${escapeHtml(src.snippet || src.content || '').slice(0, 420)}</div>
        </div>
      `).join('');
    }

    async function loadLineage(term) {
      const target = $('lineageContent');
      if (!target || !state.workspace) return;
      try {
        const res = await fetch(`/api/lineage?workspace=${encodeURIComponent(state.workspace)}&term=${encodeURIComponent(term)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const relations = data.relations || [];
        if (!relations.length) {
          target.innerHTML = `<p>${escapeHtml(data.summary || 'No citation-backed relationship was resolved for this term.')}</p>`;
          return;
        }
        target.className = '';
        target.innerHTML = `
          <p>${escapeHtml(data.summary || '')}</p>
          <div class="lineage-list">
            ${relations.map(item => {
              const context = (item.contexts || [])[0];
              const title = escapeHtml(item.title || `Reference ${item.reference_number}`);
              const linkedTitle = item.url
                ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${title}</a>`
                : title;
              return `
                <div class="lineage-card ${escapeHtml(item.relation || 'background')}">
                  <div class="lineage-head">
                    <div class="lineage-title">[${escapeHtml(item.reference_number)}] ${linkedTitle}${item.year ? ` · ${escapeHtml(item.year)}` : ''}</div>
                    <span class="tag relation-badge">${escapeHtml((item.relation || 'background').replace('_', ' '))} · ${escapeHtml(item.confidence || 'low')}</span>
                  </div>
                  <div class="subtle">${escapeHtml(item.explanation || '')}</div>
                  ${context ? `<div class="citation-context">${formatInline(context.text || '')} <button class="page-link" data-page-target="${escapeHtml(context.page)}">Page ${escapeHtml(context.page)}</button></div>` : ''}
                </div>`;
            }).join('')}
          </div>`;
      } catch (err) {
        target.innerHTML = `<p>Lineage unavailable: ${escapeHtml(err.message)}</p>`;
      }
    }

    function delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    async function enrichmentFetch(url, options={}) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 12000);
      try {
        return await fetch(url, {...options, signal: controller.signal});
      } finally {
        clearTimeout(timer);
      }
    }

    function showMathRenderFallback(container) {
      const rendered = container.querySelector('.math-rendered');
      if (!rendered) return;
      const page = rendered.dataset.page || '';
      rendered.className = 'math-render-fallback';
      rendered.textContent = `A reliable equation preview could not be produced${page ? ` for page ${page}` : ''}. The original notation remains visible in the paper; extraction details are available below.`;
    }

    function replaceInlineMathErrors(container) {
      container.querySelectorAll('.inline-math-source').forEach(wrapper => {
        if (!wrapper.querySelector('[data-mml-node="merror"], mjx-merror')) return;
        const latex = wrapper.dataset.latex || wrapper.textContent || '';
        const fallback = document.createElement('code');
        fallback.className = 'inline-math-fallback';
        fallback.textContent = readableInlineMath(latex) || latex;
        wrapper.replaceWith(fallback);
      });
    }

    function typesetMath(container, attempt=0) {
      if (window.MathJax?.typesetPromise) {
        window.MathJax.typesetPromise([container]).then(() => {
          const rendered = container.querySelector('.math-rendered');
          if (rendered?.querySelector('[data-mml-node="merror"], mjx-merror')) showMathRenderFallback(container);
          replaceInlineMathErrors(container);
        }).catch(error => {
          console.error('MathJax failed', error);
          showMathRenderFallback(container);
          container.querySelectorAll('.inline-math-source').forEach(wrapper => {
            const fallback = document.createElement('code');
            fallback.className = 'inline-math-fallback';
            fallback.textContent = readableInlineMath(wrapper.dataset.latex || wrapper.textContent);
            wrapper.replaceWith(fallback);
          });
        });
      } else if (attempt < 12) {
        setTimeout(() => typesetMath(container, attempt + 1), 250);
      }
    }

    function renderMathPopover(explanation, mathKey) {
      const pop = $('popover');
      if (pop.dataset.mathKey !== mathKey) return;
      const parsed = explanation.parse || {};
      const symbols = explanation.symbols || [];
      const steps = explanation.steps || [];
      const paperEvidence = explanation.paper_evidence || [];
      const assumptions = explanation.assumptions_or_missing_details || [];
      const derivationNotes = explanation.derivation_notes || '';
      const toyExample = explanation.toy_example || '';
      const transcriptionWarnings = explanation.transcription_warnings || [];
      const evaluation = explanation.evaluation || {};
      const mathLoop = explanation.loop || {};
      const loopIterations = Array.isArray(mathLoop.iterations) ? mathLoop.iterations : [];
      const dimensions = evaluation.dimensions || {};
      const dimensionLabels = {
        correctness: 'Correctness',
        paper_grounding: 'Grounding',
        symbol_coverage: 'Symbols',
        latex_fidelity: 'LaTeX',
        usefulness: 'Usefulness',
      };
      const dimensionEntries = Object.entries(dimensionLabels).filter(([key]) => dimensions[key]);
      const verdict = evaluation.verdict || 'review';
      const displayLatex = String(explanation.display_latex || '').trim();
      pop.innerHTML = `
        <div class="popover-header">
          <div>
            <h3>${explanation.region_kind === 'inline' ? 'Inline math' : 'Equation'} · page ${escapeHtml(explanation.page)}</h3>
            <div class="tag-row">
              <span class="tag">math-walkthrough</span><span class="tag">SymPy</span><span class="tag">paper-math-agent</span>
            </div>
          </div>
          <button id="closePop">Close</button>
        </div>
        <h4>Equation</h4>
        ${displayLatex
          ? `<div class="math-rendered" data-page="${escapeHtml(explanation.page)}">\\[${escapeHtml(displayLatex)}\\]</div>`
          : `<div class="math-render-fallback">A reliable equation preview could not be produced for page ${escapeHtml(explanation.page)}. The original notation remains visible in the paper; extraction details are available below.</div>`}
        <h4>Role</h4>
        <p class="role-copy">${escapeHtml(humanizeLabel(explanation.role))}</p>
        <h4>Plain English</h4>
        <p>${formatInline(explanation.plain_english || '')}</p>
        <h4>Step by step</h4>
        <ol class="evidence">${steps.map(item => cleanListItem(item)).filter(Boolean).map(item => `<li>${formatInline(item)}</li>`).join('') || '<li>No steps were produced.</li>'}</ol>
        <h4>Symbols</h4>
        ${symbols.length ? `
          <table class="symbol-table"><thead><tr><th>Symbol</th><th>Meaning</th><th>Basis</th></tr></thead><tbody>
          ${symbols.map(item => `<tr><td><span class="math-symbol inline-math-source" data-latex="${escapeHtml(item.symbol)}">\\(${escapeHtml(item.symbol)}\\)</span></td><td>${formatInline(item.meaning)}</td><td><span class="symbol-source">${escapeHtml(humanizeLabel(item.source, 'Unresolved'))}</span></td></tr>`).join('')}
          </tbody></table>` : '<p class="subtle">No symbols were resolved.</p>'}
        <h4>Intuition</h4>
        <p>${formatInline(explanation.intuition || 'No intuition was produced.')}</p>
        <h4>Dimensions and shapes</h4>
        <p>${formatTechnicalInline(explanation.dimensional_analysis || 'Dimensions were not resolved.')}</p>
        <h4>Implementation view</h4>
        <div class="implementation-view">${renderImplementation(explanation.implementation_view)}</div>
        <h4>Derivation notes</h4>
        <p>${formatTechnicalInline(derivationNotes || 'The paper does not provide a derivation for this equation.')}</p>
        <h4>Toy example</h4>
        <p>${formatTechnicalInline(toyExample || 'No faithful toy example was produced.')}</p>
        <h4>How it fits the paper</h4>
        <p>${formatInline(explanation.context_fit || '')}</p>
        <h4>Paper evidence</h4>
        <ul class="evidence paper-evidence">${paperEvidence.map(item => cleanPaperEvidence(item)).filter(Boolean).map(item => `<li>${formatTechnicalInline(item)}</li>`).join('') || '<li>No supporting paper evidence was produced.</li>'}</ul>
        ${assumptions.length ? `
          <h4>Assumptions or missing details</h4>
          <ul class="evidence">${assumptions.map(item => cleanListItem(item)).filter(Boolean).map(item => `<li>${formatInline(item)}</li>`).join('')}</ul>
        ` : ''}
        <details class="technical-details">
          <summary>Technical details</summary>
          <div class="subtle">${escapeHtml(explanation.model)} · ${escapeHtml(explanation.status)} · confidence: ${escapeHtml(explanation.explanation_confidence || 'low')} · ${escapeHtml(parsed.status || 'unparsed')} · LaTeX: ${escapeHtml(explanation.latex_source || 'unavailable')} (${escapeHtml(String(Math.round((Number(explanation.transcription_confidence) || 0) * 100)))}%)</div>
          ${transcriptionWarnings.length ? `<h4>Transcription warnings</h4><ul class="evidence">${transcriptionWarnings.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
          <h4>Nearby paper text</h4>
          <p>${formatInline(explanation.paper_context || 'No nearby explanation was extracted.')}</p>
          <h4>Extracted PDF text</h4>
          <div class="equation-card">${escapeHtml(explanation.raw_equation || '')}</div>
          <h4>Deterministic parse</h4>
          <div class="equation-card">${escapeHtml(parsed.sympy_form || parsed.normalized || 'SymPy could not parse the extracted notation.')}</div>
          ${parsed.error ? `<p class="subtle">Parser note: ${escapeHtml(parsed.error)}</p>` : ''}
        </details>
        ${loopIterations.length ? `
          <details class="trace-details">
            <summary>Explanation loop · ${escapeHtml(String(mathLoop.initialScore ?? 0))} → ${escapeHtml(String(mathLoop.finalScore ?? 0))}/5</summary>
            <p class="subtle">Stopped: ${escapeHtml(String(mathLoop.stopReason || 'unknown').replaceAll('_', ' '))}. ${escapeHtml(String(mathLoop.acceptedRepairs ?? 0))} repair(s) accepted from a budget of ${escapeHtml(String(mathLoop.attemptBudget ?? 0))}.</p>
            <div class="judge-reasons">
              ${loopIterations.map(item => `
                <div class="judge-reason">
                  <b>${item.iteration === 0 ? 'Initial answer' : `Repair ${escapeHtml(item.iteration)}`}:</b>
                  ${escapeHtml(item.verdict || 'review')} · ${escapeHtml(item.score ?? 0)}/5 ·
                  ${escapeHtml(item.action || '')}
                  ${(item.appliedFeedback || []).length
                    ? `<div class="subtle">Applied: ${(item.appliedFeedback || []).map(feedback => escapeHtml(feedback)).join(' ')}</div>`
                    : ''}
                </div>
              `).join('')}
            </div>
          </details>
        ` : ''}
        <details class="technical-details judge-details">
          <summary>Quality check · ${escapeHtml(verdict)} · ${escapeHtml(evaluation.overall_score ?? 0)}/5</summary>
          <div class="judge-panel">
          <div class="judge-header">
            <div><b>${escapeHtml(evaluation.summary || 'Evaluation unavailable.')}</b><div class="subtle">${escapeHtml(evaluation.judge_model || 'No judge model')}</div></div>
            <span class="judge-verdict ${escapeHtml(verdict)}">${escapeHtml(verdict)} · ${escapeHtml(evaluation.overall_score ?? 0)}/5</span>
          </div>
          ${dimensionEntries.length ? `<div class="judge-grid">${dimensionEntries.map(([key, label]) => `
            <div class="judge-score"><b>${escapeHtml(dimensions[key].score)}/5</b><span>${escapeHtml(label)}</span></div>
          `).join('')}</div>` : ''}
          <div class="judge-reasons">${dimensionEntries.map(([key, label]) => `
            <div class="judge-reason"><b>${escapeHtml(label)}:</b> ${formatInline(dimensions[key].justification || '')}</div>
          `).join('')}</div>
          ${(evaluation.issues || []).length ? `<ul class="evidence">${evaluation.issues.map(item => `<li>${formatInline(item)}</li>`).join('')}</ul>` : ''}
            <div class="judge-feedback">
              <div class="judge-feedback-actions">
                <button data-math-judge-rating="agree">Judge looks right</button>
                <button data-math-judge-rating="disagree">Correct the judge</button>
              </div>
              <div class="subtle" data-math-judge-feedback-status></div>
              <div class="judge-correction">
                <select aria-label="Judge dimension">
                  <option value="overall">Overall judgment</option>
                  <option value="correctness">Correctness</option>
                  <option value="paper_grounding">Paper grounding</option>
                  <option value="symbol_coverage">Symbol coverage</option>
                  <option value="latex_fidelity">LaTeX fidelity</option>
                  <option value="usefulness">Usefulness</option>
                </select>
                <textarea maxlength="4000" placeholder="What rule should the judge apply next time?" aria-label="Judge correction"></textarea>
                <label class="memory-check"><input type="checkbox" checked /> Use this guidance for future equations</label>
                <button data-save-math-judge-feedback>Save judge guidance</button>
              </div>
            </div>
            <details class="trace-details"><summary>Deterministic evaluation checks</summary><div class="equation-card">${escapeHtml(JSON.stringify(evaluation.deterministic_checks || {}, null, 2))}</div></details>
          </div>
        </details>
      `;
      $('closePop').onclick = () => {
        pop.classList.remove('open');
        delete pop.dataset.mathKey;
      };
      if (displayLatex) typesetMath(pop);
    }

    async function openMath(page, regionId) {
      const layout = state.pageLayouts.get(Number(page));
      const region = layout?.mathRegions?.find(item => item.id === regionId);
      if (!region) return;
      const mathKey = `${state.workspace}::${page}::${regionId}`;
      const pop = $('popover');
      delete pop.dataset.termKey;
      pop.dataset.mathKey = mathKey;
      pop.classList.add('open');
      if (state.mathExplanations.has(mathKey)) {
        renderMathPopover(state.mathExplanations.get(mathKey), mathKey);
        return;
      }
      pop.innerHTML = `
        <div class="popover-header"><h3>${region.kind === 'inline' ? 'Inline math' : 'Equation'} · page ${escapeHtml(page)}</h3><button id="closePop">Close</button></div>
        <div class="math-render-fallback">Reading the original equation and reconstructing its notation...</div>
        <p id="mathStatus" class="enrich-status">Preparing SymPy and paper-agent analysis...</p>
      `;
      $('closePop').onclick = () => {
        pop.classList.remove('open');
        delete pop.dataset.mathKey;
      };
      if (state.mathPending.has(mathKey)) return;
      state.mathPending.add(mathKey);
      try {
        const start = await enrichmentFetch(
          `/api/math-explain?workspace=${encodeURIComponent(state.workspace)}&page=${encodeURIComponent(page)}&region=${encodeURIComponent(regionId)}`,
          {method: 'POST'},
        );
        if (!start.ok) throw new Error(await start.text());
        let job = await start.json();
        const deadline = Date.now() + 255000;
        while (job.status === 'running' && Date.now() < deadline) {
          const status = $('mathStatus');
          if (status && pop.dataset.mathKey === mathKey) status.textContent = `${job.message} (${Math.round(job.elapsedSeconds || 0)}s)`;
          await delay(1200);
          const poll = await enrichmentFetch(`/api/enrichment-status?job=${encodeURIComponent(job.jobId)}`);
          if (!poll.ok) throw new Error(await poll.text());
          job = await poll.json();
        }
        const explanation = job.result?.math;
        if (job.status !== 'completed' || !explanation) throw new Error(job.message || 'Math analysis timed out.');
        state.mathExplanations.set(mathKey, explanation);
        renderMathPopover(explanation, mathKey);
        loadContextDiagnostics().catch(() => {});
      } catch (err) {
        const status = $('mathStatus');
        if (status && pop.dataset.mathKey === mathKey) status.textContent = `Math analysis stopped: ${err.name === 'AbortError' ? 'request timed out' : err.message}`;
      } finally {
        state.mathPending.delete(mathKey);
      }
    }

    async function enrichTerm(term) {
      const status = $('enrichStatus');
      const autoKey = `${state.workspace}::${term.toLowerCase()}`;
      const isCurrentTerm = () => $('popover').dataset.termKey === autoKey;
      if (status && isCurrentTerm()) status.textContent = 'Starting Tavily Research...';
      try {
        const start = await enrichmentFetch(
          `/api/enrich-term?workspace=${encodeURIComponent(state.workspace)}&term=${encodeURIComponent(term)}`,
          {method: 'POST'},
        );
        if (!start.ok) throw new Error(await start.text());
        let job = await start.json();
        const deadline = Date.now() + 255000;
        while (job.status === 'running' && Date.now() < deadline) {
          const currentStatus = $('enrichStatus');
          if (currentStatus && isCurrentTerm()) currentStatus.textContent = `${job.message} (${Math.round(job.elapsedSeconds || 0)}s)`;
          await delay(1500);
          const poll = await enrichmentFetch(`/api/enrichment-status?job=${encodeURIComponent(job.jobId)}`);
          if (!poll.ok) throw new Error(await poll.text());
          job = await poll.json();
        }
        if (job.status === 'completed') {
          if ($('popover').classList.contains('open') && isCurrentTerm()) await openTerm(term);
          loadContextDiagnostics().catch(() => {});
          return;
        }
        throw new Error(job.message || 'Enrichment timed out. Open the term to retry.');
      } catch (err) {
        autoWebAttempted.delete(autoKey);
        const currentStatus = $('enrichStatus');
        if (currentStatus && isCurrentTerm()) currentStatus.textContent = `Enrichment stopped: ${err.name === 'AbortError' ? 'request timed out' : err.message}`;
      }
    }

    function cleanSelectedTerm(text) {
      return String(text || '').replace(/\s+/g, ' ').trim().replace(/^[,.;:()\[\]{}]+|[,.;:()\[\]{}]+$/g, '');
    }

    function capturePaperSelection() {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed || !selection.rangeCount) return;
      const term = cleanSelectedTerm(selection.toString());
      if (term.length < 2 || term.length > 120) return;
      const range = selection.getRangeAt(0);
      const node = range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
        ? range.commonAncestorContainer
        : range.commonAncestorContainer.parentElement;
      const pageShell = node?.closest?.('.pdf-page-shell');
      if (!pageShell) return;
      state.selectionPage = Number(pageShell.dataset.page);
      $('manualTerm').value = term;
      $('selectionLabel').textContent = `Remember “${term}”`;
      const rect = range.getBoundingClientRect();
      const action = $('selectionAction');
      action.style.left = `${Math.max(12, Math.min(window.innerWidth - 352, rect.left))}px`;
      action.style.top = `${Math.max(12, Math.min(window.innerHeight - 112, rect.bottom + 8))}px`;
      action.classList.add('open');
    }

    async function rememberHighlight(term, category, page=null) {
      const clean = cleanSelectedTerm(term);
      if (!clean) {
        $('memoryStatus').textContent = 'Select or enter a term first.';
        return;
      }
      $('memoryStatus').textContent = `Remembering ${clean}...`;
      $('rememberBtn').disabled = true;
      $('selectionSave').disabled = true;
      try {
        const res = await fetch(`/api/remember-term?workspace=${encodeURIComponent(state.workspace)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({term: clean, category, page}),
        });
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        renderState(payload.state);
        $('manualTerm').value = clean;
        $('manualCategory').value = category;
        $('memoryStatus').textContent = `${clean} is now learned across papers.`;
        $('selectionAction').classList.remove('open');
        window.getSelection()?.removeAllRanges();
      } catch (err) {
        $('memoryStatus').textContent = err.message;
      } finally {
        $('rememberBtn').disabled = false;
        $('selectionSave').disabled = false;
      }
    }

    document.addEventListener('click', (event) => {
      const sidePanel = event.target.closest('[data-side-panel]');
      if (sidePanel) {
        setSidePanel(sidePanel.dataset.sidePanel);
        return;
      }
      const modeTarget = event.target.closest('[data-chat-mode]');
      if (modeTarget) {
        state.chatMode = modeTarget.dataset.chatMode;
        document.querySelectorAll('[data-chat-mode]').forEach(button => button.classList.toggle('active', button === modeTarget));
        return;
      }
      const chatPage = event.target.closest('[data-chat-page]');
      if (chatPage) {
        document.getElementById(`pdf-page-${chatPage.dataset.chatPage}`)?.scrollIntoView({behavior: 'smooth', block: 'start'});
        return;
      }
      const chatUrl = event.target.closest('[data-chat-url]');
      if (chatUrl) {
        window.open(chatUrl.dataset.chatUrl, '_blank', 'noopener,noreferrer');
        return;
      }
      const ratingButton = event.target.closest('[data-chat-rating]');
      if (ratingButton) {
        const messageElement = ratingButton.closest('.chat-message');
        const rating = ratingButton.dataset.chatRating;
        if (rating === 'not_helpful') {
          messageElement.querySelector('.chat-correction')?.classList.add('open');
        } else {
          submitChatFeedback(messageElement, rating).then(() => {
            ratingButton.parentElement.innerHTML = '<span class="subtle">Feedback saved.</span>';
          }).catch(err => {
            $('chatStatus').textContent = err.message;
          });
        }
        return;
      }
      const correctionButton = event.target.closest('[data-save-correction]');
      if (correctionButton) {
        const messageElement = correctionButton.closest('.chat-message');
        const form = correctionButton.closest('.chat-correction');
        const correction = form.querySelector('textarea').value.trim();
        const remember = form.querySelector('input[type=checkbox]').checked;
        const status = form.querySelector('[data-feedback-status]');
        if (!correction) {
          status.textContent = 'Enter a correction first.';
          return;
        }
        correctionButton.disabled = true;
        submitChatFeedback(messageElement, 'not_helpful', correction, remember).then(() => {
          status.textContent = remember ? 'Correction saved to long-term chat memory.' : 'Correction saved for this paper.';
        }).catch(err => {
          status.textContent = err.message;
        }).finally(() => {
          correctionButton.disabled = false;
        });
        return;
      }
      const judgeRating = event.target.closest('[data-math-judge-rating]');
      if (judgeRating) {
        const feedbackRoot = judgeRating.closest('.judge-feedback');
        if (judgeRating.dataset.mathJudgeRating === 'disagree') {
          feedbackRoot.querySelector('.judge-correction')?.classList.add('open');
        } else {
          const status = feedbackRoot.querySelector('[data-math-judge-feedback-status]');
          judgeRating.disabled = true;
          submitMathJudgeFeedback('agree').then(() => {
            status.textContent = 'Judge agreement saved as a calibration example.';
            judgeRating.parentElement.innerHTML = '<span class="subtle">Judge agreement saved.</span>';
          }).catch(err => {
            status.textContent = err.message;
          }).finally(() => {
            judgeRating.disabled = false;
          });
        }
        return;
      }
      const saveJudgeFeedback = event.target.closest('[data-save-math-judge-feedback]');
      if (saveJudgeFeedback) {
        const form = saveJudgeFeedback.closest('.judge-correction');
        const feedback = form.querySelector('textarea').value.trim();
        const dimension = form.querySelector('select').value;
        const remember = form.querySelector('input[type=checkbox]').checked;
        const status = form.querySelector('[data-math-judge-feedback-status]');
        if (!feedback) {
          status.textContent = 'Describe what the judge should do differently.';
          return;
        }
        saveJudgeFeedback.disabled = true;
        submitMathJudgeFeedback('disagree', dimension, feedback, remember).then(() => {
          status.textContent = remember
            ? 'Guidance saved to judge memory.'
            : 'Feedback saved for this paper only.';
        }).catch(err => {
          status.textContent = err.message;
        }).finally(() => {
          saveJudgeFeedback.disabled = false;
        });
        return;
      }
      const mathTarget = event.target.closest('.math-hotspot');
      if (mathTarget) {
        event.preventDefault();
        clearTimeout(mathHoverTimer);
        openMath(Number(mathTarget.dataset.mathPage), mathTarget.dataset.mathRegion);
        return;
      }
      const target = event.target.closest('[data-term]');
      if (target) openTerm(target.dataset.term);
      const pageTarget = event.target.closest('[data-page-target]');
      if (pageTarget) {
        document.getElementById(`pdf-page-${pageTarget.dataset.pageTarget}`)?.scrollIntoView({behavior: 'smooth', block: 'start'});
      }
    });
    document.addEventListener('pointerover', (event) => {
      const mathTarget = event.target.closest('.math-hotspot');
      if (!mathTarget) return;
      clearTimeout(mathHoverTimer);
      mathHoverTimer = setTimeout(() => {
        openMath(Number(mathTarget.dataset.mathPage), mathTarget.dataset.mathRegion);
      }, 700);
    });
    document.addEventListener('pointerout', (event) => {
      if (!event.target.closest('.math-hotspot')) return;
      clearTimeout(mathHoverTimer);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        $('popover').classList.remove('open');
        delete $('popover').dataset.mathKey;
      }
    });
    $('fileInput').addEventListener('change', () => {
      const file = $('fileInput').files[0];
      $('uploadStatus').textContent = file
        ? `Ready to parse ${file.name}.`
        : 'Choose a PDF, then upload and parse.';
    });
    $('uploadBtn').addEventListener('click', uploadPdf);
    $('sendChatBtn').addEventListener('click', sendChat);
    $('chatInput').addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChat();
      }
    });
    $('newChatBtn').addEventListener('click', () => {
      state.threadId = newThreadId();
      state.lastQuestion = '';
      state.chatQuestions = new Map();
      localStorage.setItem(threadStorageKey(), state.threadId);
      renderChatHistory([]);
      $('chatStatus').textContent = '';
      $('chatInput').focus();
    });
    $('refreshContext').addEventListener('click', () => {
      $('contextDebug').textContent = 'Refreshing context diagnostics...';
      loadContextDiagnostics().catch(err => {
        $('contextDebug').textContent = `Context diagnostics unavailable: ${err.message}`;
      });
    });
    $('reader').addEventListener('mouseup', () => setTimeout(capturePaperSelection, 0));
    $('reader').addEventListener('scroll', () => $('selectionAction').classList.remove('open'));
    $('rememberBtn').addEventListener('click', () => rememberHighlight(
      $('manualTerm').value,
      $('manualCategory').value,
      state.selectionPage,
    ));
    $('selectionSave').addEventListener('click', () => rememberHighlight(
      $('manualTerm').value,
      $('selectionCategory').value,
      state.selectionPage,
    ));
    loadState().catch(err => {
      $('buildLabel').textContent = BUILD_LABEL;
      $('workspaceText').textContent = 'No parsed workspace loaded';
      $('paperTitle').textContent = 'Paper workspace';
      $('paperMeta').textContent = 'Open a PDF to begin.';
      $('tavilyText').textContent = 'Waiting for a paper';
      $('reader').innerHTML = '<div class="empty"><div><strong>No paper open</strong>Choose a PDF from the sidebar to create a local research workspace.</div></div>';
      $('stats').innerHTML = '';
    });
  </script>
</body>
</html>
"""


class PaperReaderHandler(BaseHTTPRequestHandler):
    server_version = "DeepPaperReader/0.1"

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _error(self, status: int, message: str) -> None:
        self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        try:
            if parsed_url.path == "/":
                self._send(HTTPStatus.OK, APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed_url.path == "/api/health":
                self._json({"ok": True, "build": BUILD_LABEL})
                return
            if parsed_url.path == "/api/context-diagnostics":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                diagnostic_workspace: Workspace | None = None
                if workspace_arg:
                    diagnostic_workspace, _ = load_workspace(workspace_arg)
                else:
                    latest = latest_workspace()
                    if latest:
                        diagnostic_workspace = Workspace(latest)
                self._json(context_diagnostics_payload(diagnostic_workspace))
                return
            if parsed_url.path == "/api/state":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                workspace, parsed = load_workspace(workspace_arg)
                self._json(state_payload(workspace, parsed))
                return
            if parsed_url.path == "/api/term":
                query = parse_qs(parsed_url.query)
                term = query.get("term", [""])[0].strip()
                if not term:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing term")
                    return
                workspace_arg = query.get("workspace", [None])[0]
                workspace, parsed = load_workspace(workspace_arg)
                self._json(term_payload(workspace, parsed, term))
                return
            if parsed_url.path == "/api/lineage":
                query = parse_qs(parsed_url.query)
                term = query.get("term", [""])[0].strip()
                workspace_arg = query.get("workspace", [None])[0]
                workspace, parsed = load_workspace(workspace_arg)
                result = build_research_lineage(parsed, workspace, term)
                self._json(research_lineage_payload(result))
                return
            if parsed_url.path == "/api/page-layout":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                page_number = int(query.get("page", ["0"])[0])
                workspace, parsed = load_workspace(workspace_arg)
                terms = load_significant_terms(parsed, workspace, max_terms=120)
                self._json(page_layout_payload(parsed, terms, page_number))
                return
            if parsed_url.path == "/api/page-image":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                page_number = int(query.get("page", ["0"])[0])
                workspace, parsed = load_workspace(workspace_arg)
                self._send(
                    HTTPStatus.OK, render_page_png(workspace, parsed, page_number), "image/png"
                )
                return
            if parsed_url.path == "/api/enrichment-status":
                query = parse_qs(parsed_url.query)
                job_id = query.get("job", [""])[0].strip()
                if not job_id:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing enrichment job ID")
                    return
                self._json(get_enrichment_job(job_id))
                return
            if parsed_url.path == "/api/chat-status":
                query = parse_qs(parsed_url.query)
                job_id = query.get("job", [""])[0].strip()
                if not job_id:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing paper chat job ID")
                    return
                self._json(get_chat_job(job_id))
                return
            if parsed_url.path == "/api/chat-history":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                thread_id = query.get("thread", [""])[0].strip()
                if not thread_id:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing paper chat thread ID")
                    return
                workspace, _ = load_workspace(workspace_arg)
                self._json(load_chat_thread(workspace, thread_id))
                return
            self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        try:
            if parsed_url.path == "/api/enrich-term":
                query = parse_qs(parsed_url.query)
                term = query.get("term", [""])[0].strip()
                if not term:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing term")
                    return
                workspace_arg = query.get("workspace", [None])[0]
                workspace, parsed = load_workspace(workspace_arg)
                self._json(
                    start_enrichment_job(workspace, parsed, term), status=HTTPStatus.ACCEPTED
                )
                return
            if parsed_url.path == "/api/math-explain":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                page_number = int(query.get("page", ["0"])[0])
                region_id = query.get("region", [""])[0].strip()
                if not region_id:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing math region ID")
                    return
                workspace, parsed = load_workspace(workspace_arg)
                terms = load_significant_terms(parsed, workspace, max_terms=120)
                layout = page_layout_payload(parsed, terms, page_number)
                region = next(
                    (item for item in layout.get("mathRegions", []) if item.get("id") == region_id),
                    None,
                )
                if not isinstance(region, dict):
                    self._error(HTTPStatus.NOT_FOUND, "Math region not found")
                    return
                self._json(
                    start_math_job(
                        workspace,
                        parsed,
                        page_number,
                        region_id,
                        str(region.get("raw") or ""),
                        str(region.get("context") or ""),
                        str(region.get("kind") or "display"),
                        region.get("layout_evidence")
                        if isinstance(region.get("layout_evidence"), dict)
                        else None,
                    ),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed_url.path == "/api/remember-term":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing or oversized memory payload")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                term = str(payload.get("term", ""))
                category = str(payload.get("category", "concept"))
                page = payload.get("page")
                page_number = int(page) if page is not None else None
                workspace, parsed = load_workspace(workspace_arg)
                self._json(remember_highlight(workspace, parsed, term, category, page_number))
                return
            if parsed_url.path == "/api/chat":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32_768:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing or oversized paper chat payload")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                question = str(payload.get("question") or "").strip()
                mode = str(payload.get("mode") or "fast").strip().lower()
                thread_id = str(payload.get("threadId") or "").strip() or None
                if not question:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing paper chat question")
                    return
                if mode not in {"fast", "deep", "web", "explore"}:
                    self._error(HTTPStatus.BAD_REQUEST, "Unsupported paper chat mode")
                    return
                workspace, parsed = load_workspace(workspace_arg)
                self._json(
                    start_chat_job(workspace, parsed, question, mode, thread_id),
                    status=HTTPStatus.ACCEPTED,
                )
                return
            if parsed_url.path == "/api/chat-feedback":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32_768:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing or oversized paper chat feedback")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                rating = str(payload.get("rating") or "")
                if rating not in {"helpful", "not_helpful"}:
                    self._error(HTTPStatus.BAD_REQUEST, "Unsupported feedback rating")
                    return
                workspace, _ = load_workspace(workspace_arg)
                record = record_chat_feedback(
                    workspace,
                    thread_id=str(payload.get("threadId") or ""),
                    message_id=str(payload.get("messageId") or ""),
                    rating=rating,  # type: ignore[arg-type]
                    question=str(payload.get("question") or ""),
                    correction=str(payload.get("correction") or ""),
                    remember=bool(payload.get("remember")),
                )
                correction = str(payload.get("correction") or "")
                score_session(
                    paper_session_id(workspace, str(payload.get("threadId") or "")),
                    "user_helpful",
                    1.0 if rating == "helpful" else 0.0,
                    data_type="BOOLEAN",
                    comment=(
                        correction[:500]
                        if correction and capture_content()
                        else f"User marked the answer {rating.replace('_', ' ')}."
                    ),
                    metadata={
                        "messageId": str(payload.get("messageId") or ""),
                        "rememberedCorrection": bool(payload.get("remember")),
                    },
                )
                self._json({"ok": True, "feedback": record})
                return
            if parsed_url.path == "/api/math-judge-feedback":
                query = parse_qs(parsed_url.query)
                workspace_arg = query.get("workspace", [None])[0]
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65_536:
                    self._error(HTTPStatus.BAD_REQUEST, "Missing or oversized judge feedback")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                rating = str(payload.get("rating") or "")
                if rating not in {"agree", "disagree"}:
                    self._error(HTTPStatus.BAD_REQUEST, "Unsupported judge feedback rating")
                    return
                explanation = payload.get("explanation")
                if not isinstance(explanation, dict) or not explanation.get("display_latex"):
                    self._error(HTTPStatus.BAD_REQUEST, "Missing equation explanation")
                    return
                feedback = str(payload.get("feedback") or "").strip()
                if rating == "disagree" and not feedback:
                    self._error(HTTPStatus.BAD_REQUEST, "A judge correction is required")
                    return
                workspace, _ = load_workspace(workspace_arg)
                record = record_math_judge_feedback(
                    workspace,
                    explanation=explanation,
                    rating=rating,  # type: ignore[arg-type]
                    dimension=str(payload.get("dimension") or "overall"),
                    feedback=feedback,
                    remember=bool(payload.get("remember", True)),
                )
                score_session(
                    paper_session_id(workspace),
                    "math_judge_human_agreement",
                    1.0 if rating == "agree" else 0.0,
                    data_type="BOOLEAN",
                    comment=(
                        feedback[:500]
                        if feedback and capture_content()
                        else f"User marked the math judge {rating}."
                    ),
                    metadata={
                        "equationId": str(explanation.get("equation_id") or ""),
                        "dimension": str(payload.get("dimension") or "overall"),
                        "remembered": bool(payload.get("remember", True)),
                    },
                )
                self._json({"ok": True, "feedback": record})
                return
            if parsed_url.path != "/api/upload":
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            query = parse_qs(parsed_url.query)
            filename = query.get("filename", ["paper.pdf"])[0] or "paper.pdf"
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._error(HTTPStatus.BAD_REQUEST, "Empty upload")
                return
            data = self.rfile.read(length)
            workspace, parsed = parse_uploaded_pdf(filename, data)
            self._json(state_payload(workspace, parsed))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args)
        )


def main() -> None:
    port = DEFAULT_PORT
    server = ThreadingHTTPServer(("localhost", port), PaperReaderHandler)
    print(f"Deep Paper Reader running at http://localhost:{port}")
    try:
        server.serve_forever()
    finally:
        flush_observability()


if __name__ == "__main__":
    main()

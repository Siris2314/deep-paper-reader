"""Document-bound evidence records and cached hybrid retrieval.

Coordinates are PDF points, never guessed from character offsets. Semantic retrieval
defaults to a small Hugging Face model running locally, with an Ollama adapter available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import tempfile
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import fitz

from paper_agent.workspace import Workspace

_CACHE: dict[str, list[dict]] = {}
_LOCK = threading.RLock()
_MODELS: dict[tuple, object] = {}
_MODEL_ERRORS: dict[tuple, tuple[float, str]] = {}


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold())


@lru_cache(maxsize=64)
def _pdf_digest(path: str, size: int, modified: int, changed: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


def document_id(parsed) -> str:
    source = Path(parsed.metadata.source_pdf)
    if source.is_file():
        stamp = source.stat()
        return _pdf_digest(
            str(source.resolve()), stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns
        )
    return hashlib.sha256(parsed.full_text.encode()).hexdigest()[:24]


def _ocr_layouts(workspace):
    layouts, stamps = {}, []
    if not workspace:
        return layouts, stamps
    try:
        statuses = json.loads(workspace.read_text("parsed/ocr_status.json"))
    except (OSError, ValueError):
        return layouts, stamps
    for number, status in statuses.items():
        if status != "ocr" or not str(number).isdigit():
            continue
        path = workspace.path(f"parsed/ocr_layout/{int(number):04d}.json")
        try:
            content = path.read_bytes()
            layouts[int(number)] = json.loads(content).get("dict", {})
            stamps.append((number, hashlib.sha256(content).hexdigest()))
        except (OSError, ValueError, AttributeError):
            continue
    return layouts, stamps


def _section_heading(text: str) -> str | None:
    heading = " ".join(text.split()).strip()
    if re.fullmatch(r"(?:\d+[. ]+)?(?:references|bibliography)", heading, re.I):
        return "references"
    if len(heading) <= 120 and (
        re.match(r"^(?:appendix|appendices)\b", heading, re.I)
        or re.match(r"^[A-Z](?:\.|\s)\s*[A-Z][a-z]", heading)
    ):
        return "appendix"
    return None


def build_evidence_records(parsed, workspace: Workspace | None = None) -> list[dict]:
    source = Path(parsed.metadata.source_pdf)
    stamp = source.stat() if source.is_file() else None
    ocr_layouts, ocr_stamps = _ocr_layouts(workspace)
    structure = hashlib.sha256(
        json.dumps(
            [
                [c.model_dump() for c in parsed.equation_cards],
                [c.model_dump() for c in parsed.figure_cards],
                [c.model_dump() for c in parsed.table_cards],
                parsed.section_markers,
                ocr_stamps,
                hashlib.sha256(parsed.full_text.encode()).hexdigest(),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    key = f"evidence-v7:{document_id(parsed)}:{source.resolve()}:{stamp.st_mtime_ns if stamp else 0}:{structure}"
    with _LOCK:
        if key in _CACHE:
            return [dict(item) for item in _CACHE[key]]
    if workspace and workspace.path("parsed/evidence_index.json").exists():
        try:
            cached = json.loads(workspace.read_text("parsed/evidence_index.json"))
            if cached.get("key") == key:
                with _LOCK:
                    _CACHE[key] = cached["records"]
                return [dict(item) for item in cached["records"]]
        except (OSError, ValueError, KeyError):
            pass
    records: list[dict] = []
    doc = fitz.open(source) if source.is_file() else None
    try:
        section = "body"
        page_sections = {}
        for page_number, text in sorted(parsed.page_text.items()):
            page = doc[page_number - 1] if doc and page_number <= len(doc) else None
            blocks = []
            ocr_mode = page_number in ocr_layouts
            if ocr_mode:
                for block in ocr_layouts[page_number].get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    body = "\n".join(
                        "".join(str(s.get("text", "")) for s in line.get("spans", []))
                        for line in block.get("lines", [])
                    )
                    bbox = block.get("bbox")
                    if body.strip():
                        blocks.append(
                            (body.strip(), list(bbox) if bbox and len(bbox) == 4 else None)
                        )
            elif page:
                for block in page.get_text("blocks", sort=True):
                    if len(block) > 6 and block[6] == 0 and str(block[4]).strip():
                        blocks.append((str(block[4]).strip(), list(block[:4])))
                # A scanned body can coexist with a native header. Do not discard the
                # OCR text merely because a small native block is present.
                if sum(len(body) for body, _ in blocks) < len(text) * 0.5:
                    blocks = []
                    ocr_mode = True
            if not blocks:
                blocks = [
                    (part.strip(), None) for part in re.split(r"\n\s*\n", text) if part.strip()
                ]
            for index, (body, bbox) in enumerate(blocks, 1):
                section = _section_heading(body) or section
                page_sections.setdefault(page_number, section)
                # Retain complete blocks; chunk long blocks with distinct IDs and the same
                # honest source region rather than inventing more precise coordinates.
                for offset in range(0, len(body), 1800):
                    suffix = f".{offset // 1800}" if offset else ""
                    records.append(
                        dict(
                            id=f"P{page_number}.{index}{suffix}",
                            source="paper",
                            kind="page",
                            text=body[offset : offset + 1800],
                            page=page_number,
                            bbox=bbox,
                            page_width=page.rect.width if page else None,
                            page_height=page.rect.height if page else None,
                            document_id=document_id(parsed),
                            provenance="uploaded_pdf_ocr" if ocr_mode else "uploaded_pdf",
                            section=section,
                        )
                    )
            if page and not ocr_mode:
                from paper_agent.math_regions import extract_math_regions

                for region in extract_math_regions(
                    page.get_text("dict", sort=True), page_number, page.rect.width, page.rect.height
                ):
                    if region["kind"] != "display":
                        continue
                    records.append(
                        dict(
                            id=f"EQ:{region['id']}",
                            title=f"Equation {region.get('equation_label')}"
                            if region.get("equation_label")
                            else region["id"],
                            source="paper",
                            kind="equation",
                            text=region["raw"],
                            page=page_number,
                            bbox=region["layout_evidence"]["bbox"],
                            page_width=page.rect.width,
                            page_height=page.rect.height,
                            document_id=document_id(parsed),
                            provenance="uploaded_pdf",
                            section=next(
                                (
                                    r["section"]
                                    for r in records
                                    if r.get("page") == page_number
                                    and r.get("bbox")
                                    and r["bbox"][1] <= region["y"] <= r["bbox"][3]
                                ),
                                page_sections.get(page_number, section),
                            ),
                        )
                    )
        for kind, cards, prefix in (
            ("equation", parsed.equation_cards, "EQ"),
            ("table", parsed.table_cards, "TABLE"),
            ("figure", parsed.figure_cards, "FIG"),
        ):
            for card in cards:
                text = getattr(card, "raw", "") or "\n".join(
                    filter(
                        None,
                        [
                            getattr(card, "caption", ""),
                            getattr(card, "raw_text", ""),
                            getattr(card, "surrounding_text", ""),
                        ],
                    )
                )
                page_number = card.page
                page = (
                    doc[page_number - 1]
                    if doc and page_number and page_number <= len(doc)
                    else None
                )
                probe = getattr(card, "caption", "") or text
                hits = page.search_for(probe[:100]) if page and probe else []
                section_match = next(
                    (
                        r
                        for r in records
                        if r.get("page") == page_number
                        and r["kind"] == "page"
                        and probe[:80] in r["text"]
                    ),
                    None,
                )
                records.append(
                    dict(
                        id=f"{prefix}:{card.id}",
                        source="paper",
                        kind=kind,
                        text=text,
                        title=getattr(card, "caption", "") or card.id,
                        page=page_number,
                        bbox=list(hits[0]) if hits else None,
                        page_width=page.rect.width if page else None,
                        page_height=page.rect.height if page else None,
                        document_id=document_id(parsed),
                        provenance="uploaded_pdf",
                        section=section_match["section"]
                        if section_match
                        else page_sections.get(page_number, "body"),
                        # Caption coordinates do not assert the visual's location.
                        location_kind="caption" if hits and kind in {"figure", "table"} else "text",
                    )
                )
    finally:
        if doc:
            doc.close()
    with _LOCK:
        if len(_CACHE) > 12:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = records
    if workspace:
        workspace.write_json(
            "parsed/evidence_index.json", {"version": 3, "key": key, "records": records}
        )
    return [dict(item) for item in records]


def resolve_query(question: str, history: list[dict] | None = None) -> str:
    if not history:
        return question
    if len(tokens(question)) <= 7 or re.search(
        r"\b(it|that|this|they|those|their|above)\b", question, re.I
    ):
        previous = [
            str(item.get("content") or "") for item in history if item.get("role") == "user"
        ]
        if previous:
            return f"{previous[-1][:600]}\nFollow-up: {question}"
    return question


def _embed(texts: list[str], model: str, *, query: bool = False) -> list[list[float]]:
    provider = os.getenv("PAPER_RETRIEVAL_EMBEDDING_PROVIDER", "huggingface")
    if provider == "huggingface":
        device = os.getenv("PAPER_RETRIEVAL_DEVICE", "cpu")
        revision = os.getenv("PAPER_RETRIEVAL_MODEL_REVISION") or None
        key = (model, device, revision)
        with _LOCK:
            if key in _MODEL_ERRORS and time.monotonic() - _MODEL_ERRORS[key][0] < 120:
                raise RuntimeError(_MODEL_ERRORS[key][1])
            if key not in _MODELS:
                try:
                    from sentence_transformers import SentenceTransformer

                    _MODELS[key] = SentenceTransformer(
                        model, device=device, revision=revision, trust_remote_code=False
                    )
                except Exception as exc:
                    _MODEL_ERRORS[key] = (time.monotonic(), type(exc).__name__)
                    raise
            encoder = _MODELS[key]
            values = texts
            if query and "bge-" in model.casefold():
                values = [
                    "Represent this sentence for searching relevant passages: " + text
                    for text in texts
                ]
            return encoder.encode(
                values, batch_size=24, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
    if provider != "ollama":
        raise ValueError("Unknown embedding provider")
    endpoint = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/embed"
    payload = json.dumps({"model": model, "input": texts, "truncate": True}).encode()
    with urllib.request.urlopen(
        urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        ),
        timeout=30,
    ) as response:
        data = json.load(response)
    vectors = data.get("embeddings", [])
    if len(vectors) != len(texts) or not all(isinstance(v, list) and v for v in vectors):
        raise ValueError("Embedding provider returned incomplete vectors.")
    return vectors


def _cosine(a, b):
    if len(a) != len(b):
        return 0.0
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def _validate_vectors(vectors, count: int, dimensions: int | None = None):
    if not isinstance(vectors, list) or len(vectors) != count or not vectors:
        raise ValueError("Embedding count mismatch")
    width = dimensions or (len(vectors[0]) if isinstance(vectors[0], list) else 0)
    for vector in vectors:
        if not isinstance(vector, list) or not width or len(vector) != width:
            raise ValueError("Embedding dimension mismatch")
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
            for v in vector
        ):
            raise ValueError("Embedding contains invalid values")
        if not any(v != 0 for v in vector):
            raise ValueError("Embedding has zero norm")
    return vectors


def _atomic_cache(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _similarity_threshold() -> float:
    try:
        value = float(os.getenv("PAPER_RETRIEVAL_MIN_SIMILARITY", "0.6"))
        return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.6
    except ValueError:
        return 0.6


def evidence_allowed(record: dict, query: str) -> bool:
    if record.get("section") == "references":
        return bool(re.search(r"\b(references|bibliography|citations?|cited|cites)\b", query, re.I))
    if record.get("section") == "appendix" and re.search(
        r"\b(main|contributions?|overview|summari[sz]e|summary)\b", query, re.I
    ):
        return bool(re.search(r"\b(appendix|appendices|proof|derivation)\b", query, re.I))
    return True


def search_records(
    records: list[dict], query: str, limit: int = 12, workspace: Workspace | None = None
) -> tuple[list[dict], dict]:
    records = [record for record in records if evidence_allowed(record, query)]
    if not records:
        return [], {"mode": "bm25", "candidates": 0}
    query_tokens = set(tokens(query)) - {
        "the",
        "a",
        "an",
        "is",
        "what",
        "how",
        "this",
        "that",
        "of",
        "in",
        "which",
        "why",
        "was",
        "were",
        "are",
        "does",
        "do",
        "did",
        "to",
        "for",
        "and",
        "or",
        "by",
        "with",
        "its",
        "it",
        "from",
        "on",
        "as",
        "be",
        "been",
        "used",
        "use",
        "paper",
        "explain",
        "can",
        "could",
        "would",
        "should",
        "there",
        "their",
        "about",
        "these",
        "those",
    }
    counts = [Counter(tokens(item["text"])) for item in records]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    frequency = Counter(term for count in counts for term in count)
    scores = []
    for index, count in enumerate(counts):
        score = 0.0
        for term in query_tokens:
            tf = count[term]
            if tf:
                idf = math.log(1 + (len(records) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / average))
        scores.append(score)
    lexical = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    fused = {i: 1 / (60 + rank) for rank, i in enumerate(lexical, 1) if scores[i] > 0}
    diagnostics = {"mode": "bm25", "candidates": len(records), "embedding_cache_hit": False}
    model = os.getenv("PAPER_RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
    provider = os.getenv("PAPER_RETRIEVAL_EMBEDDING_PROVIDER", "huggingface")
    if model and workspace and provider != "off":
        try:
            digest = hashlib.sha256(
                json.dumps(
                    [
                        provider,
                        model,
                        os.getenv("PAPER_RETRIEVAL_MODEL_REVISION"),
                        [(r["id"], r["text"]) for r in records],
                    ]
                ).encode()
            ).hexdigest()[:20]
            cache = workspace.path(f"retrieval/embeddings-{digest}.json")
            with _LOCK:
                vectors = None
                if cache.exists():
                    try:
                        vectors = _validate_vectors(
                            json.loads(cache.read_text(encoding="utf-8")), len(records)
                        )
                        diagnostics["embedding_cache_hit"] = True
                    except (OSError, ValueError, TypeError):
                        diagnostics["embedding_cache_rebuilt"] = True
                if vectors is None:
                    vectors = []
                    for start in range(0, len(records), 24):
                        vectors.extend(
                            _embed([r["text"] for r in records[start : start + 24]], model)
                        )
                    _validate_vectors(vectors, len(records))
                    _atomic_cache(cache, vectors)
            q = _validate_vectors(_embed([query], model, query=True), 1, len(vectors[0]))[0]
            dense = sorted(range(len(records)), key=lambda i: _cosine(q, vectors[i]), reverse=True)
            diagnostics["top_semantic_similarity"] = round(_cosine(q, vectors[dense[0]]), 4)
            for rank, i in enumerate(dense[: max(limit * 4, 40)], 1):
                if _cosine(q, vectors[i]) >= _similarity_threshold():
                    fused[i] = fused.get(i, 0) + 1 / (60 + rank)
            diagnostics["mode"] = "hybrid_rrf"
            diagnostics["embedding_model"] = model
        except Exception as exc:
            diagnostics["embedding_error"] = type(exc).__name__
    # Exact artifact references outrank fuzzy matching, including late-paper artifacts.
    reference = re.search(r"\b(eq(?:uation)?|fig(?:ure)?|table)\.?\s*\(?([0-9]+)\)?", query, re.I)
    if diagnostics.get("mode") == "hybrid_rrf" and not reference:
        lexical_support = max((len(query_tokens & set(count)) for count in counts), default=0)
        threshold = _similarity_threshold()
        if diagnostics.get("top_semantic_similarity", 0) < threshold and lexical_support < 2:
            fused = {}
            diagnostics["abstention_reason"] = "weak_semantic_and_lexical_support"
    explicit_pages = re.findall(r"\bpage\s+(\d+)\b", query, re.I)
    for i, item in enumerate(records):
        if explicit_pages and item.get("page") == int(explicit_pages[-1]):
            fused[i] = fused.get(i, 0) + 0.05
        if reference:
            kind = {
                "eq": "equation",
                "equation": "equation",
                "fig": "figure",
                "figure": "figure",
                "table": "table",
            }[reference[1].lower()]
            number = int(reference[2])
            own = re.search(r"(\d+)\D*$", item["id"])
            label = re.search(rf"\b{reference[1]}\.?\s*{number}\b", item.get("title", ""), re.I)
            printed = re.search(r"\(\s*" + str(number) + r"\s*\)\s*[,.;]?\s*$", item["text"])
            if item["kind"] == kind and (
                label or printed or ("math-" not in item["id"] and own and int(own[1]) == number)
            ):
                fused[i] = fused.get(i, 0) + 1
        if item.get("section") == "appendix" and not re.search(
            r"\b(appendix|proof|derive|derivation)\b", query, re.I
        ):
            if i in fused:
                fused[i] *= 0.7
    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)
    reranker = os.getenv("PAPER_RETRIEVAL_RERANKER", "").strip()
    if reranker and ranked:
        try:
            from sentence_transformers import CrossEncoder

            key = ("reranker", reranker)
            with _LOCK:
                if key not in _MODELS:
                    _MODELS[key] = CrossEncoder(
                        reranker,
                        device=os.getenv("PAPER_RETRIEVAL_DEVICE", "cpu"),
                        trust_remote_code=False,
                    )
                pool = ranked[: min(30, len(ranked))]
                reranked = _MODELS[key].predict([(query, records[i]["text"]) for i in pool])
            # Explicit artifact references remain pinned above reranked prose.
            ranked = sorted(
                pool, key=lambda i: (fused[i] >= 1, float(reranked[pool.index(i)])), reverse=True
            )
            diagnostics["reranker"] = reranker
        except Exception as exc:
            diagnostics["reranker_error"] = type(exc).__name__
    # No positive match is an observable retrieval miss, not arbitrary first pages.
    return [dict(records[i], score=round(fused[i], 6)) for i in ranked[:limit]], diagnostics


def render_source_crop(parsed, page: int, bbox=None, scale: float = 1.5) -> bytes:
    with fitz.open(parsed.metadata.source_pdf) as doc:
        if page < 1 or page > len(doc):
            raise ValueError("Page is outside the document.")
        source = doc[page - 1]
        rect = source.rect
        if bbox is not None:
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(math.isfinite(float(v)) for v in bbox)
            ):
                raise ValueError("Invalid source rectangle.")
            rect = fitz.Rect(*bbox) & source.rect
            if rect.is_empty or rect.width < 2 or rect.height < 2:
                raise ValueError("Source rectangle is empty.")
        return source.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False).tobytes(
            "png"
        )

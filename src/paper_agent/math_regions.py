from __future__ import annotations

import re
from typing import Any

from paper_agent.parser import clean_line, looks_like_equation

MATH_FONT_HINTS = ("math", "cmmi", "cmsy", "cmex", "symbol")
EXPLANATION_HINTS = re.compile(
    r"\b(where|denote|denotes|represent|represents|defined|definition|given|computed|corresponds|"
    r"respectively|every|range|scale|projection|weight|probability|score|loss|objective)\b",
    re.I,
)
MATH_CHARS = re.compile(r"[=+*/^<>∈∑Σ∏πτθλμσρφψωΔ×·≤≥≈]")


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _union(rects: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _is_math_span(span: dict[str, Any]) -> bool:
    font = str(span.get("font") or "").lower()
    text = str(span.get("text") or "")
    return any(hint in font for hint in MATH_FONT_HINTS) or bool(MATH_CHARS.search(text))


def _rect_payload(
    rect: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    pad_x: float = 3,
    pad_y: float = 2,
) -> dict[str, float]:
    x0 = max(0.0, rect[0] - pad_x)
    y0 = max(0.0, rect[1] - pad_y)
    x1 = min(page_width, rect[2] + pad_x)
    y1 = min(page_height, rect[3] + pad_y)
    return {
        "x": round(x0, 3),
        "y": round(y0, 3),
        "width": round(max(2.0, x1 - x0), 3),
        "height": round(max(2.0, y1 - y0), 3),
    }


def _visual_rows(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for record in sorted(records, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        placed = False
        for row in rows:
            row_box = _union([item["bbox"] for item in row])
            overlap = min(row_box[3], record["bbox"][3]) - max(row_box[1], record["bbox"][1])
            min_height = min(row_box[3] - row_box[1], record["bbox"][3] - record["bbox"][1])
            if overlap >= -2 or overlap >= min_height * 0.25:
                row.append(record)
                placed = True
                break
        if not placed:
            rows.append([record])
    return rows


def _layout_evidence(group: list[dict[str, Any]]) -> dict[str, object]:
    """Keep the PDF geometry needed to recover scripts lost by plain-text extraction."""
    spans: list[dict[str, object]] = []
    for item in group:
        for span in item["spans"]:
            rect = _bbox(span.get("bbox"))
            text = str(span.get("text") or "")
            if not rect or not text.strip():
                continue
            spans.append(
                {
                    "text": text.strip(),
                    "x": round(rect[0], 2),
                    "top": round(rect[1], 2),
                    "bottom": round(rect[3], 2),
                    "size": round(float(span.get("size") or 0), 2),
                    "font": str(span.get("font") or ""),
                    "source_block": int(item["block"]),
                    "source_line": int(item["line"]),
                }
            )
    spans.sort(
        key=lambda span: (
            int(span["source_block"]),
            int(span["source_line"]),
            float(span["x"]),
        )
    )
    group_box = _union([item["bbox"] for item in group])
    return {
        "coordinate_note": (
            "PDF y coordinates increase downward. Smaller glyphs above a neighboring base are superscripts; "
            "smaller glyphs below it are subscripts. Very small glyphs may be nested scripts."
        ),
        "spans": spans,
        "geometry_latex": _geometry_latex(spans),
        "bbox": [round(value, 2) for value in group_box],
    }


def _geometry_latex(spans: list[dict[str, object]]) -> str:
    if not spans:
        return ""

    def center(span: dict[str, object]) -> float:
        return (float(span["top"]) + float(span["bottom"])) / 2

    def normalize(span: dict[str, object]) -> str:
        text = str(span["text"]).strip()
        font = str(span.get("font") or "").casefold()
        if "extension" in font and text in {"X", "∑", "Σ"}:
            return r"\sum"
        if "extension" in font and text == "[":
            return r"\bigcup"
        if "msbm" in font and text == "I":
            return r"\mathbb{I}"
        if "roman" in font and len(text) > 1 and text.isalpha():
            return rf"\mathrm{{{text}}}"
        replacements = {
            "·": r"\cdot",
            "∈": r"\in ",
            "≤": r"\le ",
            "≥": r"\ge ",
            "−": "-",
            "τ": r"\tau",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text.replace(". . .", r"\ldots")

    max_size = max(float(span["size"]) for span in spans)
    main_spans = [span for span in spans if float(span["size"]) >= max_size * 0.85]
    script_spans = [span for span in spans if float(span["size"]) < max_size * 0.85]
    if not main_spans:
        return ""

    row_groups: list[list[dict[str, object]]] = []
    for span in sorted(main_spans, key=lambda item: (center(item), float(item["x"]))):
        target_row = next(
            (
                row
                for row in row_groups
                if abs(center(span) - sum(center(item) for item in row) / len(row))
                <= max_size * 0.45
            ),
            None,
        )
        if target_row is None:
            row_groups.append([span])
        else:
            target_row.append(span)

    rendered_rows: list[str] = []
    script_count = 0
    for row_spans in sorted(row_groups, key=lambda row: sum(center(item) for item in row) / len(row)):
        baseline = sorted(center(span) for span in row_spans)[len(row_spans) // 2]
        atoms = [
            {
                "base": normalize(span),
                "x": float(span["x"]),
                "sup": "",
                "sub": "",
            }
            for span in sorted(row_spans, key=lambda item: float(item["x"]))
        ]
        nearby_scripts = [
            span
            for span in script_spans
            if abs(center(span) - baseline) <= max_size * 1.7
        ]
        for span in sorted(nearby_scripts, key=lambda item: float(item["x"])):
            script_x = float(span["x"])
            candidates = [
                atom
                for atom in atoms
                if (
                    float(atom["x"]) <= script_x + 3
                    or (
                        str(atom["base"]) in {r"\sum", r"\bigcup"}
                        and abs(float(atom["x"]) - script_x) <= 12
                    )
                )
                and re.search(r"[A-Za-z0-9}\]]$", str(atom["base"]))
            ]
            if not candidates:
                continue
            target = min(candidates, key=lambda atom: abs(script_x - float(atom["x"])))
            position = "sup" if center(span) < baseline - max_size * 0.08 else "sub"
            target[position] = str(target[position]) + normalize(span)
            script_count += 1

        rendered: list[str] = []
        for atom in atoms:
            base = str(atom["base"])
            sub = str(atom["sub"])
            sup = str(atom["sup"])
            if sub:
                base += f"_{{{sub}}}"
            if sup:
                base += f"^{{{sup}}}"
            rendered.append(base)
        row = " ".join(rendered)
        row = re.sub(r"\s+([),.;])", r"\1", row)
        row = re.sub(r"([([])\s+", r"\1", row)
        row = re.sub(r"\\in\s+M", r"\\in M", row)
        rendered_rows.append(row)

    if not rendered_rows or script_count == 0 or not any("=" in row for row in rendered_rows):
        return ""
    latex = (
        rendered_rows[0]
        if len(rendered_rows) == 1
        else r"\begin{aligned} "
        + r" \\ ".join(row.replace(" = ", " &= ", 1) for row in rendered_rows)
        + r" \end{aligned}"
    )
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", latex):
        return ""
    return latex


def extract_math_regions(
    page_dict: dict[str, Any], page_number: int, page_width: float, page_height: float
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    by_block: dict[int, list[dict[str, Any]]] = {}
    for block_index, block in enumerate(page_dict.get("blocks", [])):
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            if not isinstance(line, dict):
                continue
            bbox = _bbox(line.get("bbox"))
            spans = [span for span in line.get("spans", []) if isinstance(span, dict)]
            if not bbox or not spans:
                continue
            text = clean_line("".join(str(span.get("text") or "") for span in spans))
            if not text:
                continue
            total_width = max(1.0, bbox[2] - bbox[0])
            math_width = sum(
                max(0.0, span_box[2] - span_box[0])
                for span in spans
                if _is_math_span(span) and (span_box := _bbox(span.get("bbox")))
            )
            max_size = max(float(span.get("size") or 0) for span in spans)
            operator_count = len(MATH_CHARS.findall(text))
            record = {
                "block": block_index,
                "line": line_index,
                "bbox": bbox,
                "spans": spans,
                "text": text,
                "math_ratio": math_width / total_width,
                "max_size": max_size,
                "operator_count": operator_count,
            }
            lines.append(record)
            by_block.setdefault(block_index, []).append(record)

    display_groups: list[list[dict[str, Any]]] = []
    claimed_lines: set[tuple[int, int]] = set()
    candidates: list[dict[str, Any]] = []
    for line in lines:
        explanatory_prose = (
            bool(EXPLANATION_HINTS.search(line["text"]))
            and len(re.findall(r"[A-Za-z]{2,}", line["text"])) >= 3
        )
        is_seed = line["max_size"] >= 6 and (
            (line["math_ratio"] >= 0.3 and (not explanatory_prose or line["math_ratio"] >= 0.75))
            or (
                looks_like_equation(line["text"])
                and len(line["text"]) <= 100
                and not explanatory_prose
            )
            or (
                line["operator_count"] >= 3
                and line["math_ratio"] >= 0.2
                and len(line["text"]) <= 180
                and not explanatory_prose
            )
        )
        if is_seed:
            candidates.append(line)

    # PDF generators often emit brackets, bases, and subscripts as separate lines. Pull short
    # neighboring components into the seed set before grouping logical equations.
    candidate_keys = {(item["block"], item["line"]) for item in candidates}
    for block_lines in by_block.values():
        seeds = [item for item in block_lines if (item["block"], item["line"]) in candidate_keys]
        for line in block_lines:
            key = (line["block"], line["line"])
            if key in candidate_keys or len(line["text"]) > 80 or line["max_size"] < 5:
                continue
            if line["math_ratio"] < 0.12 and line["operator_count"] == 0:
                continue
            if any(
                abs(line["bbox"][1] - seed["bbox"][1]) <= 9
                or abs(line["bbox"][3] - seed["bbox"][3]) <= 9
                for seed in seeds
            ):
                candidates.append(line)
                candidate_keys.add(key)

    candidate_rows = _visual_rows(candidates)
    candidate_rows.sort(
        key=lambda row: (
            _union([item["bbox"] for item in row])[1],
            _union([item["bbox"] for item in row])[0],
        )
    )
    for row in candidate_rows:
        if not display_groups:
            display_groups.append(list(row))
            continue
        previous_box = _union([item["bbox"] for item in display_groups[-1]])
        current_box = _union([item["bbox"] for item in row])
        vertical_gap = current_box[1] - previous_box[3]
        vertical_overlap = min(previous_box[3], current_box[3]) - max(
            previous_box[1], current_box[1]
        )
        horizontally_near = (
            current_box[0] <= previous_box[2] + 55 and current_box[2] >= previous_box[0] - 55
        )
        if horizontally_near and (vertical_gap <= 7 or vertical_overlap >= -2):
            display_groups[-1].extend(row)
        else:
            display_groups.append(list(row))
    for group in display_groups:
        claimed_lines.update((item["block"], item["line"]) for item in group)

    regions: list[dict[str, Any]] = []
    sorted_lines = sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for group in display_groups:
        rows = _visual_rows(group)
        row_rects = [_union([item["bbox"] for item in row]) for row in rows]
        raw_rows = [
            "".join(
                item["text"] for item in sorted(row, key=lambda value: value["bbox"][0])
            ).strip()
            for row in rows
        ]
        group_box = _union(row_rects)
        nearby = [
            item["text"]
            for item in sorted_lines
            if item not in group
            and item["bbox"][1] >= group_box[1] - 130
            and item["bbox"][3] <= group_box[3] + 100
        ]
        region_id = f"math-{page_number}-display-{len(regions) + 1}"
        regions.append(
            {
                "id": region_id,
                "kind": "display",
                "raw": "\n".join(row for row in raw_rows if row),
                "context": "\n".join(nearby[:14]),
                "layout_evidence": _layout_evidence(group),
                "rects": [_rect_payload(rect, page_width, page_height) for rect in row_rects],
                **_rect_payload(group_box, page_width, page_height),
            }
        )

    for line in sorted_lines:
        if (line["block"], line["line"]) in claimed_lines or line["max_size"] < 7:
            continue
        spans = line["spans"]
        math_indices = [index for index, span in enumerate(spans) if _is_math_span(span)]
        if not math_indices:
            continue
        if (
            not EXPLANATION_HINTS.search(line["text"])
            and line["operator_count"] == 0
            and len(math_indices) < 2
        ):
            continue
        runs: list[list[int]] = []
        explanatory_line = bool(EXPLANATION_HINTS.search(line["text"]))
        for index in math_indices:
            intervening = ""
            previous_text = ""
            if runs:
                previous_text = str(spans[runs[-1][-1]].get("text") or "")
                intervening = "".join(
                    str(spans[pos].get("text") or "") for pos in range(runs[-1][-1] + 1, index)
                )
            same_clause = (
                explanatory_line
                and not re.search(r"[,;]", intervening)
                and not re.search(r"[,;]\s*$", previous_text)
            )
            if runs and (index <= runs[-1][-1] + 2 or same_clause):
                runs[-1].append(index)
            else:
                runs.append([index])
        for run in runs:
            start, end = min(run), max(run)
            while start > 0:
                previous = str(spans[start - 1].get("text") or "").strip()
                if re.fullmatch(r"[A-Za-z0-9_\[\](,;:]", previous):
                    start -= 1
                else:
                    break
            while end + 1 < len(spans):
                following = str(spans[end + 1].get("text") or "").strip()
                if re.fullmatch(r"[A-Za-z0-9_\]\[),;:]", following):
                    end += 1
                else:
                    break
            selected = spans[start : end + 1]
            selected_rects = [rect for span in selected if (rect := _bbox(span.get("bbox")))]
            raw = "".join(str(span.get("text") or "") for span in selected).strip()
            if not raw or not selected_rects or not re.search(r"[A-Za-z0-9α-ωΑ-Ω=<>∈∑Σ]", raw):
                continue
            rect = _union(selected_rects)
            region_id = f"math-{page_number}-inline-{len(regions) + 1}"
            line_position = sorted_lines.index(line)
            context = "\n".join(
                item["text"] for item in sorted_lines[max(0, line_position - 2) : line_position + 3]
            )
            regions.append(
                {
                    "id": region_id,
                    "kind": "inline",
                    "raw": raw,
                    "context": context,
                    "rects": [_rect_payload(rect, page_width, page_height, pad_x=5, pad_y=3)],
                    **_rect_payload(rect, page_width, page_height, pad_x=5, pad_y=3),
                }
            )
    return regions

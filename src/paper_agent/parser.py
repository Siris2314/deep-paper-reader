from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from paper_agent.schemas import EquationCard, FigureCard, PaperMetadata, TableCard
from paper_agent.workspace import Workspace

SECTION_PATTERNS = [
    r"^\s*(abstract)\s*$",
    r"^\s*(introduction)\s*$",
    r"^\s*(related work)\s*$",
    r"^\s*(background)\s*$",
    r"^\s*(method|methods|methodology|approach|model|architecture)\s*$",
    r"^\s*(experiments|experimental setup|evaluation|results)\s*$",
    r"^\s*(discussion)\s*$",
    r"^\s*(limitations|limitations and future work|limitations and diagnostics)\s*$",
    r"^\s*(conclusion|conclusions)\s*$",
    r"^\s*(references|bibliography)\s*$",
    r"^\s*\d+\.?\s+([A-Z][A-Za-z0-9 ,:;\-()]+)\s*$",
]

EQUATION_LINE_HINTS = [
    "=",
    "∑",
    "Σ",
    "∈",
    "≤",
    "≥",
    "σ",
    "log",
    "exp",
    "softmax",
    "BCE",
    "loss",
    "argmax",
    "argmin",
]

CAPTION_RE = re.compile(r"\b(Figure|Table)\s+(\d+)\b[:\.\s-]*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedPaper:
    metadata: PaperMetadata
    full_text: str
    page_text: dict[int, str]
    sections: dict[str, str]
    equation_cards: list[EquationCard]
    figure_cards: list[FigureCard]
    table_cards: list[TableCard]

    @property
    def equation_candidates(self) -> list[str]:
        return [card.raw for card in self.equation_cards]


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def extract_pdf_text(pdf_path: str | Path) -> tuple[str, dict[int, str], PaperMetadata]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    page_text: dict[int, str] = {}
    full_parts: list[str] = []

    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        text = text.strip()
        page_text[idx] = text
        full_parts.append(f"\n\n--- PAGE {idx} ---\n{text}")

    full_text = "\n".join(full_parts).strip()
    metadata = PaperMetadata(
        title_guess=guess_title(full_text),
        authors_guess=guess_authors(full_text),
        page_count=len(doc),
        source_pdf=str(pdf_path),
    )
    return full_text, page_text, metadata


def guess_title(text: str) -> str | None:
    first_page = text.split("--- PAGE 2 ---")[0]
    lines = [clean_line(line) for line in first_page.splitlines()]
    lines = [line for line in lines if line and not line.startswith("--- PAGE")]

    # Many arXiv papers wrap titles across 2-3 lines. Join the first contiguous title-ish block.
    title_lines: list[str] = []
    for line in lines[:8]:
        if re.search(r"@|University|Tencent|Princeton|arXiv|Abstract", line, re.I):
            break
        if len(line) >= 5:
            title_lines.append(line)
        if len(" ".join(title_lines)) > 20 and len(title_lines) >= 2:
            break
    if title_lines:
        return " ".join(title_lines).strip()

    for line in lines[:20]:
        if 8 <= len(line) <= 180 and not re.match(r"^(abstract|introduction)$", line, re.I):
            return line
    return None


def guess_authors(text: str) -> list[str]:
    first_page = text.split("--- PAGE 2 ---")[0]
    lines = [clean_line(line) for line in first_page.splitlines()]
    candidates: list[str] = []
    for line in lines[1:18]:
        if "@" in line or re.search(
            r"University|Tencent|Independent|Abstract|Date:|Correspondence:|^We\s+introduce|^Conventional|^Progress",
            line,
            re.I,
        ):
            break
        if "," in line and len(line) < 350:
            candidates.append(line)
    if not candidates:
        return []
    merged = " ".join(candidates)
    merged = re.sub(r"\d+|∗|\*|†|,", " ", merged)
    names = [clean_line(x) for x in re.split(r"\s{2,}| and ", merged) if clean_line(x)]
    return names[:30]


def detect_section_heading(line: str) -> str | None:
    if not line or len(line) > 120:
        return None
    for pattern in SECTION_PATTERNS:
        match = re.match(pattern, line, flags=re.I)
        if match:
            if match.groups():
                return match.group(1)
            return line
    return None


def split_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {"front_matter": []}
    current = "front_matter"

    for line in lines:
        normalized = clean_line(line)
        section_name = detect_section_heading(normalized)
        if section_name:
            current = section_name.lower().replace(" ", "_").replace("/", "_")
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def extract_equation_cards(
    page_text: dict[int, str], max_candidates: int = 120
) -> list[EquationCard]:
    cards: list[EquationCard] = []
    eq_id = 1

    # Capture local windows around explicit equation numbers like (4), plus math-heavy lines.
    numbered_pattern = re.compile(r"(.{0,500}?\(\d{1,3}\))")

    for page, text in page_text.items():
        compact = re.sub(r"\n+", "\n", text)
        for match in numbered_pattern.finditer(compact):
            raw = clean_line(match.group(1))
            if looks_like_equation(raw):
                cards.append(EquationCard(id=f"eq_{eq_id:03d}", page=page, raw=raw))
                eq_id += 1
                if len(cards) >= max_candidates:
                    return cards

        lines = [clean_line(line) for line in text.splitlines()]
        for line in lines:
            if looks_like_equation(line) and not any(line in card.raw for card in cards):
                cards.append(EquationCard(id=f"eq_{eq_id:03d}", page=page, raw=line))
                eq_id += 1
                if len(cards) >= max_candidates:
                    return cards

    return cards


def looks_like_equation(line: str) -> bool:
    if len(line) < 5 or len(line) > 600:
        return False
    if re.search(r"https?://|@|ISBN", line):
        return False
    if re.search(r"\([0-9]{1,3}\)$", line) and any(h in line for h in EQUATION_LINE_HINTS):
        return True
    math_symbols = sum(1 for h in EQUATION_LINE_HINTS if h in line)
    if math_symbols >= 2:
        return True
    if re.search(r"[A-Za-z][A-Za-z0-9_,]*\s*=", line) and len(line.split()) <= 18:
        return True
    return False


def extract_captions(page_text: dict[int, str]) -> tuple[list[FigureCard], list[TableCard]]:
    figures: list[FigureCard] = []
    tables: list[TableCard] = []
    fig_seen: set[tuple[int, int]] = set()
    table_seen: set[tuple[int, int]] = set()

    for page, text in page_text.items():
        lines = [line for line in text.splitlines() if clean_line(line)]
        for i, line in enumerate(lines):
            normalized = clean_line(line)
            if re.match(r"^Figure\s+\d+", normalized, re.I):
                caption_lines = [normalized]
                for nxt in lines[i + 1 : i + 4]:
                    n = clean_line(nxt)
                    if re.match(r"^(Figure|Table)\s+\d+|^\d+\s+[A-Z]", n):
                        break
                    caption_lines.append(n)
                caption = clean_line(" ".join(caption_lines))
                num = int(re.search(r"Figure\s+(\d+)", caption, re.I).group(1))
                key = (page, num)
                if key not in fig_seen:
                    fig_seen.add(key)
                    figures.append(
                        FigureCard(
                            id=f"figure_{num}",
                            page=page,
                            caption=caption,
                            surrounding_text="\n".join(lines[max(0, i - 5) : i + 8]),
                        )
                    )
            if re.match(r"^Table\s+\d+", normalized, re.I):
                caption_lines = [normalized]
                raw_lines = []
                for nxt in lines[i + 1 : i + 20]:
                    n = clean_line(nxt)
                    if re.match(r"^\d+(\.\d+)?\s+[A-Z][A-Za-z]", n) or re.match(
                        r"^Figure\s+\d+", n, re.I
                    ):
                        break
                    raw_lines.append(n)
                    if len(caption_lines) < 5 and not re.match(r"^[\d\.\(\)%]+", n):
                        caption_lines.append(n)
                num = int(re.search(r"Table\s+(\d+)", normalized, re.I).group(1))
                key = (page, num)
                if key not in table_seen:
                    table_seen.add(key)
                    tables.append(
                        TableCard(
                            id=f"table_{num}",
                            page=page,
                            caption=clean_line(" ".join(caption_lines)),
                            raw_text="\n".join(raw_lines),
                        )
                    )

    return figures, tables


def parse_paper(pdf_path: str | Path, output_dir: str | Path) -> ParsedPaper:
    workspace = Workspace(output_dir)
    full_text, page_text, metadata = extract_pdf_text(pdf_path)
    sections = split_sections(full_text)
    equation_cards = extract_equation_cards(page_text)
    figure_cards, table_cards = extract_captions(page_text)

    parsed = ParsedPaper(
        metadata=metadata,
        full_text=full_text,
        page_text=page_text,
        sections=sections,
        equation_cards=equation_cards,
        figure_cards=figure_cards,
        table_cards=table_cards,
    )
    save_parsed_paper(parsed, workspace)
    from paper_agent.scholarly_metadata import save_local_identity

    save_local_identity(parsed, workspace)
    return parsed


def save_parsed_paper(parsed: ParsedPaper, workspace: Workspace) -> None:
    workspace.write_text("raw/extracted_text.md", parsed.full_text)
    workspace.write_text(
        "parsed/page_text.json",
        json.dumps(parsed.page_text, indent=2, ensure_ascii=False),
    )
    workspace.write_text(
        "parsed/metadata.json",
        parsed.metadata.model_dump_json(indent=2),
    )
    workspace.write_text(
        "parsed/sections.json",
        json.dumps(parsed.sections, indent=2, ensure_ascii=False),
    )
    workspace.write_text(
        "parsed/equations.json",
        json.dumps(
            [card.model_dump() for card in parsed.equation_cards], indent=2, ensure_ascii=False
        ),
    )
    workspace.write_text(
        "parsed/figures.json",
        json.dumps(
            [card.model_dump() for card in parsed.figure_cards], indent=2, ensure_ascii=False
        ),
    )
    workspace.write_text(
        "parsed/tables.json",
        json.dumps(
            [card.model_dump() for card in parsed.table_cards], indent=2, ensure_ascii=False
        ),
    )

    # Backward-compatible flat files for old scripts.
    workspace.write_text("extracted_text.md", parsed.full_text)
    workspace.write_text("metadata.json", parsed.metadata.model_dump_json(indent=2))
    workspace.write_text("sections.json", json.dumps(parsed.sections, indent=2, ensure_ascii=False))
    workspace.write_text(
        "equation_candidates.json",
        json.dumps([card.raw for card in parsed.equation_cards], indent=2, ensure_ascii=False),
    )


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED: increase --max-chars to include more paper text.]"


def parsed_summary_for_prompt(parsed: ParsedPaper, max_chars: int) -> str:
    section_names = ", ".join(parsed.sections.keys())
    equation_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.raw[:300]}" for card in parsed.equation_cards[:25]
    )
    figure_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.caption[:300]}" for card in parsed.figure_cards[:10]
    )
    table_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.caption[:300]}" for card in parsed.table_cards[:10]
    )
    paper_text = truncate_text(parsed.full_text, max_chars=max_chars)

    return f"""
METADATA
Title guess: {parsed.metadata.title_guess}
Authors guess: {", ".join(parsed.metadata.authors_guess) if parsed.metadata.authors_guess else "unknown"}
Page count: {parsed.metadata.page_count}
Detected sections: {section_names}

DETECTED FIGURES
{figure_preview or "No figure captions detected."}

DETECTED TABLES
{table_preview or "No table captions detected."}

EQUATION CANDIDATES
{equation_preview or "No equation candidates detected."}

PAPER TEXT
{paper_text}
""".strip()


def parsed_artifact_manifest(parsed: ParsedPaper) -> dict[str, Any]:
    return {
        "metadata": parsed.metadata.model_dump(),
        "sections": list(parsed.sections.keys()),
        "equation_count": len(parsed.equation_cards),
        "figure_count": len(parsed.figure_cards),
        "table_count": len(parsed.table_cards),
    }


def load_parsed_paper_from_workspace(workspace: Workspace) -> ParsedPaper:
    """Rehydrate ParsedPaper from parsed/raw workspace artifacts."""
    metadata_path = workspace.path("parsed/metadata.json")
    sections_path = workspace.path("parsed/sections.json")
    page_text_path = workspace.path("parsed/page_text.json")
    equations_path = workspace.path("parsed/equations.json")
    figures_path = workspace.path("parsed/figures.json")
    tables_path = workspace.path("parsed/tables.json")
    raw_path = workspace.path("raw/extracted_text.md")

    missing = [
        str(p.relative_to(workspace.root))
        for p in [metadata_path, sections_path, page_text_path, raw_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Workspace is missing parsed paper artifacts. Run parse first. Missing: "
            + ", ".join(missing)
        )

    metadata = PaperMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    page_payload = json.loads(page_text_path.read_text(encoding="utf-8"))
    page_text = {int(k): v for k, v in page_payload.items()}
    equations = (
        [
            EquationCard.model_validate(x)
            for x in json.loads(equations_path.read_text(encoding="utf-8"))
        ]
        if equations_path.exists()
        else []
    )
    figures = (
        [FigureCard.model_validate(x) for x in json.loads(figures_path.read_text(encoding="utf-8"))]
        if figures_path.exists()
        else []
    )
    tables = (
        [TableCard.model_validate(x) for x in json.loads(tables_path.read_text(encoding="utf-8"))]
        if tables_path.exists()
        else []
    )
    full_text = raw_path.read_text(encoding="utf-8", errors="replace")

    return ParsedPaper(
        metadata=metadata,
        full_text=full_text,
        page_text=page_text,
        sections=sections,
        equation_cards=equations,
        figure_cards=figures,
        table_cards=tables,
    )

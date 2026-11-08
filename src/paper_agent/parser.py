from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF

from paper_agent.ocr import extract_page_text
from paper_agent.schemas import EquationCard, FigureCard, PaperMetadata, TableCard
from paper_agent.structure_parser import parse_structure
from paper_agent.workspace import Workspace

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


@dataclass
class ParsedPaper:
    metadata: PaperMetadata
    full_text: str
    page_text: dict[int, str]
    sections: dict[str, str]
    equation_cards: list[EquationCard]
    figure_cards: list[FigureCard]
    table_cards: list[TableCard]
    section_markers: list[dict[str, object]] = field(default_factory=list)

    @property
    def equation_candidates(self) -> list[str]:
        return [card.raw for card in self.equation_cards]


def clean_line(line: str) -> str:
    return " ".join(line.split())


def extract_pdf_text(
    pdf_path: str | Path,
    on_ocr_page: Callable[[int, str, dict[str, Any] | None], None] | None = None,
) -> tuple[str, dict[int, str], PaperMetadata]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_text: dict[int, str] = {}
    full_parts: list[str] = []

    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            text, layout, status = extract_page_text(page)
            page_text[idx] = text
            full_parts.append(f"\n\n--- PAGE {idx} ---\n{text}")
            if on_ocr_page is not None and status is not None:
                on_ocr_page(idx, status, layout)
        page_count = len(doc)

    full_text = "\n".join(full_parts).strip()
    metadata = PaperMetadata(
        title_guess=None,
        authors_guess=[],
        page_count=page_count,
        source_pdf=str(pdf_path),
    )
    return full_text, page_text, metadata


def looks_like_equation(line: str) -> bool:
    if len(line) < 5 or len(line) > 600:
        return False
    if any(token in line.lower() for token in ("http://", "https://", "@", "isbn")):
        return False
    math_symbols = sum(1 for h in EQUATION_LINE_HINTS if h in line)
    if math_symbols >= 2:
        return True
    return "=" in line and any(char.isalpha() for char in line) and len(line.split()) <= 18


def parse_paper(
    pdf_path: str | Path, output_dir: str | Path, *, structure_model: Any | None = None
) -> ParsedPaper:
    workspace = Workspace(output_dir)
    ocr_status: dict[str, str] = {}

    def save_ocr_page(page: int, status: str, layout: dict[str, Any] | None) -> None:
        ocr_status[str(page)] = status
        if layout is not None:
            workspace.write_json(f"parsed/ocr_layout/{page:04d}.json", layout)

    full_text, page_text, metadata = extract_pdf_text(pdf_path, save_ocr_page)
    from paper_agent.harness import harness_scope

    with harness_scope(workspace, "parse"):
        structure = parse_structure(page_text, structure_model)
    metadata = metadata.model_copy(
        update={"title_guess": structure.title, "authors_guess": structure.authors}
    )

    parsed = ParsedPaper(
        metadata=metadata,
        full_text=full_text,
        page_text=page_text,
        sections=structure.sections,
        equation_cards=structure.equations,
        figure_cards=structure.figures,
        table_cards=structure.tables,
        section_markers=structure.markers,
    )
    save_parsed_paper(parsed, workspace)
    workspace.write_json("parsed/ocr_status.json", ocr_status)
    workspace.write_json("parsed/structure_status.json", structure.status)
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
    workspace.write_json("parsed/section_markers.json", parsed.section_markers)
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
    markers_path = workspace.path("parsed/section_markers.json")
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
    markers = json.loads(markers_path.read_text(encoding="utf-8")) if markers_path.exists() else []
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
        section_markers=markers,
    )

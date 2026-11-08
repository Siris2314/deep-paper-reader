import json
from pathlib import Path

import fitz
import pytest

from paper_agent import ocr
from paper_agent.parser import load_parsed_paper_from_workspace, parse_paper
from paper_agent.term_highlighter import SignificantTerm
from paper_agent.workspace import Workspace
from ui.paper_reader_server import page_layout_payload, state_payload


def _scanned_pdf(path: Path) -> Path:
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    page.insert_text((60, 140), "Sparse attention improves retrieval accuracy", fontsize=22)
    page.insert_text((60, 190), "Our method uses a compact key value cache", fontsize=18)
    page.insert_text((210, 250), "q = softmax(x)", fontsize=20)
    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
    source.close()

    scan = fitz.open()
    page = scan.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=image)
    scan.save(path)
    scan.close()
    return path


def _term() -> SignificantTerm:
    return SignificantTerm(
        term="Sparse attention",
        category="method",
        score=20,
        paper_occurrences=1,
        method_occurrences=1,
        evidence=[],
        skills=["paper-term-highlighter"],
    )


def test_scanned_page_ocr_provides_text_and_overlay(tmp_path, monkeypatch):
    tessdata = fitz.get_tessdata()
    if not tessdata or not (Path(tessdata) / "eng.traineddata").exists():
        pytest.skip("Tesseract language data is not installed")
    monkeypatch.setenv("PAPER_OCR_ENABLED", "true")
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    with fitz.open(pdf) as document:
        assert not document[0].get_text().strip()

    workspace = Workspace(tmp_path / "report")
    parsed = parse_paper(pdf, workspace.root)
    assert "Sparse attention" in parsed.page_text[1]
    assert json.loads(workspace.path("parsed/ocr_status.json").read_text()) == {"1": "ocr"}

    reloaded = load_parsed_paper_from_workspace(workspace)
    layout = page_layout_payload(reloaded, [_term()], 1, workspace)
    assert any(mark["term"] == "Sparse attention" for mark in layout["highlights"])
    assert any(word["text"].lower().startswith("sparse") for word in layout["words"])
    assert layout["mathRegions"]
    for region in layout["mathRegions"]:
        evidence = region["layout_evidence"]
        assert evidence["ocr_source"] is True
        assert not evidence["geometry_latex"]
        assert not evidence["spans"]
    assert state_payload(workspace, reloaded)["ocr"]["pages"] == 1


def test_ocr_can_be_disabled_without_changing_digital_pages(tmp_path, sample_pdf, monkeypatch):
    monkeypatch.setenv("PAPER_OCR_ENABLED", "false")
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    assert "Sparse Attention" in parsed.page_text[1]
    assert json.loads((tmp_path / "report" / "parsed" / "ocr_status.json").read_text()) == {}


def test_reparse_ignores_stale_ocr_geometry(tmp_path, sample_pdf, monkeypatch):
    monkeypatch.setenv("PAPER_OCR_ENABLED", "true")
    workspace = Workspace(tmp_path / "report")
    parse_paper(sample_pdf, workspace.root)
    stale = workspace.path("parsed/ocr_layout/0001.json")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"words": [], "dict": {}}', encoding="utf-8")
    parsed = parse_paper(sample_pdf, workspace.root)

    layout = page_layout_payload(parsed, [_term()], 1, workspace)
    assert layout["words"]
    assert any(mark["term"] == "Sparse attention" for mark in layout["highlights"])


def test_ocr_failure_keeps_original_text_and_reports_failure(tmp_path, monkeypatch):
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    monkeypatch.setenv("PAPER_OCR_ENABLED", "true")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Tesseract language data missing")

    monkeypatch.setattr(ocr, "_make_textpage", unavailable)
    workspace = Workspace(tmp_path / "report")
    parsed = parse_paper(pdf, workspace.root)
    assert not parsed.page_text[1]
    status = json.loads(workspace.path("parsed/ocr_status.json").read_text())
    assert status["1"].startswith("OCR unavailable:")
    assert state_payload(workspace, parsed)["ocr"]["unavailable"] == 1

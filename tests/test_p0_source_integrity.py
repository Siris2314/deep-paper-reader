import json
from dataclasses import replace
from pathlib import Path

import fitz
import pytest

from paper_agent.evidence_index import build_evidence_records, document_id, search_records
from paper_agent.parser import ParsedPaper, extract_pdf_text, parse_paper
from paper_agent.reader_regression import run_regression
from paper_agent.workspace import Workspace
from tests.test_ocr import _scanned_pdf


def test_document_identity_includes_pdf_bytes(sample_pdf, tmp_path):
    full, pages, metadata = extract_pdf_text(sample_pdf)
    paper = ParsedPaper(metadata, full, pages, {}, [], [], [])
    second = tmp_path / "other.pdf"
    with fitz.open(sample_pdf) as doc:
        doc.set_metadata({"subject": "different source revision"})
        doc.save(second)
    other = replace(paper, metadata=metadata.model_copy(update={"source_pdf": str(second)}))
    assert paper.full_text == other.full_text
    assert document_id(paper) != document_id(other)


def test_scanned_body_and_native_header_both_remain_retrievable(tmp_path, monkeypatch):
    tessdata = fitz.get_tessdata()
    if not tessdata or not (Path(tessdata) / "eng.traineddata").exists():
        pytest.skip("OCR language data unavailable")
    monkeypatch.setenv("PAPER_OCR_ENABLED", "true")
    path = _scanned_pdf(tmp_path / "scan.pdf")
    with fitz.open(path) as doc:
        doc[0].insert_text((40, 30), "Digitized note", fontsize=9)
        doc.saveIncr()
    workspace = Workspace(tmp_path / "report")
    paper = parse_paper(path, workspace.root)
    records = build_evidence_records(paper, workspace)
    hits, _ = search_records(records, "compact key value cache")
    assert any("cache" in h["text"] for h in hits)
    assert all(h["provenance"] == "uploaded_pdf_ocr" for h in hits)
    assert any(h["bbox"] for h in hits)
    assert not any(
        r["kind"] == "equation" and r.get("provenance") == "uploaded_pdf" for r in records
    )


@pytest.mark.parametrize(
    "corrupt", ["invalid json", "[[NaN, 0], [0, 1]]", "[[1], [0, 1]]", "[[0, 0], [0, 1]]"]
)
def test_invalid_embedding_cache_is_repaired(tmp_path, monkeypatch, corrupt):
    monkeypatch.setenv("PAPER_RETRIEVAL_EMBEDDING_PROVIDER", "huggingface")
    calls = []

    def embed(texts, model, *, query=False):
        calls.append(query)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("paper_agent.evidence_index._embed", embed)
    records = [
        {"id": "a", "kind": "page", "text": "calibrated posterior variance"},
        {"id": "b", "kind": "page", "text": "sensor hardware"},
    ]
    workspace = Workspace(tmp_path / "report")
    search_records(records, "posterior variance", workspace=workspace)
    cache = next(workspace.path("retrieval").glob("embeddings-*.json"))
    cache.write_text(corrupt, encoding="utf-8")
    hits, diagnostics = search_records(records, "posterior variance", workspace=workspace)
    assert hits
    assert diagnostics["embedding_cache_rebuilt"] is True
    assert diagnostics["mode"] == "hybrid_rrf"
    assert calls == [False, True, False, True]
    assert json.loads(cache.read_text()) == [[1.0, 0.0], [1.0, 0.0]]


def test_authored_cross_layout_regressions(tmp_path):
    corpus = json.loads(Path("llmops/reader_regression_corpus.json").read_text())
    report = run_regression(corpus, tmp_path)
    assert report["case_count"] == 30
    assert report["scientific_generalization_evaluated"] is False
    evaluated = [row for row in report["cases"] if row["status"] == "evaluated"]
    assert all(row["passed"] for row in evaluated), [r for r in evaluated if not r["passed"]]
    if report["coverage"] < 1:
        assert report["passed"] is False


def test_shared_equation_rejects_same_id_with_different_source(tmp_path):
    from paper_agent.paper_chat import RouteDecision, retrieve_paper_evidence
    from paper_agent.schemas import EquationCard
    from tests.test_math_agent import parsed_math_paper

    paper = parsed_math_paper()
    paper.equation_cards = [EquationCard(id="eq_001", page=1, raw="s = x^2 + y")]
    workspace = Workspace(tmp_path / "report")
    workspace.write_json(
        "math/artifacts/eq_001.json",
        {
            "document_id": document_id(paper),
            "page": 1,
            "raw_equation": "z = old_formula",
            "display_latex": "UNRELATED",
            "status": "agent_explained",
            "layout_evidence": {"bbox": None},
        },
    )
    route = RouteDecision(specialists=["math"], skills=[], reason="test")
    evidence = retrieve_paper_evidence(paper, "input score equation", route, workspace=workspace)
    equation = next(item for item in evidence if item.id == "EQ:eq_001")
    assert "UNRELATED" not in equation.text
    assert "Selected notation" not in equation.text


def test_reader_correction_prevents_old_symbol_explanation_reuse(tmp_path):
    from paper_agent.equation_tutor import save_correction
    from paper_agent.paper_chat import RouteDecision, retrieve_paper_evidence
    from paper_agent.schemas import EquationCard
    from tests.test_math_agent import parsed_math_paper

    paper = parsed_math_paper()
    paper.equation_cards = [EquationCard(id="eq_001", page=1, raw="s = x^2 + y")]
    workspace = Workspace(tmp_path / "report")
    workspace.write_json(
        "math/artifacts/eq_001.json",
        {
            "document_id": document_id(paper),
            "page": 1,
            "raw_equation": "s = x^2 + y",
            "display_latex": "s = x^2 + y",
            "status": "agent_explained",
            "symbols": [{"symbol": "x", "meaning": "STALE DEFINITION"}],
            "layout_evidence": {"bbox": None},
        },
    )
    save_correction(workspace, 1, "s = x^2 + y", {"bbox": None}, "s = x^3 + y")
    route = RouteDecision(specialists=["math"], skills=[], reason="test")
    evidence = retrieve_paper_evidence(paper, "input score equation", route, workspace=workspace)
    equation = next(item for item in evidence if item.id == "EQ:eq_001")
    assert "Reader-corrected notation: s = x^3 + y" in equation.text
    assert "STALE DEFINITION" not in equation.text

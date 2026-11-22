from pathlib import Path

import pytest

from paper_agent.equation_tutor import load_correction, save_correction
from paper_agent.fulltext_sources import parse_tei
from paper_agent.math_reliability_eval import _key, evaluate_math_reliability, load_math_benchmark
from paper_agent.paper_chat import ChatEvidence
from paper_agent.visual_evidence import visual_messages
from paper_agent.workspace import Workspace


def test_math_scoring_preserves_semantic_differences():
    for left, right in [("W", "w"), (r"x^{a+b}", "x^a+b"), (r"\frac{ab}{c}", r"\frac{a}{bc}")]:
        assert _key(left) != _key(right)


def test_transcription_agreement_preserves_scope_but_tolerates_style():
    from paper_agent.math_agent import _select_transcription

    valid = _select_transcription(
        parsed_latex="",
        geometry_latex=r"c_{t}^{Q}=h_{t}\cdot W^{DQ}",
        vision_latex=r"\mathbf{c}_t^Q=\mathbf{h}_t\cdot W^{DQ}",
        raw="",
        region_kind="display",
    )
    assert valid.confidence >= 0.8
    assert valid.source == "pdf_geometry"
    invalid = _select_transcription(
        parsed_latex="",
        geometry_latex="x^{a+b}",
        vision_latex="x^a+b",
        raw="",
        region_kind="display",
    )
    assert invalid.confidence < 0.6


def test_missing_measurements_are_not_reported_as_success():
    benchmark = load_math_benchmark(Path("llmops/math_reliability_benchmark.json"))
    report = evaluate_math_reliability(benchmark, [{"id": "unknown"}])
    assert report["coverage"] == 0
    assert report["metrics"]["unsupported_claim_rate"] is None
    assert report["metrics"]["median_latency_ms"] is None
    assert report["metrics"]["mean_boundary_iou"] == 0


def test_transcription_corrections_are_region_local(tmp_path):
    workspace = Workspace(tmp_path / "work")
    save_correction(workspace, 2, "x=y", {"bbox": [1, 2, 3, 4]}, "x = y")
    assert load_correction(workspace, 2, "x=y", {"bbox": [1, 2, 3, 4]})["latex"] == "x = y"
    assert load_correction(workspace, 3, "x=y", {"bbox": [1, 2, 3, 4]}) == {}
    with pytest.raises(ValueError):
        save_correction(workspace, 2, "x=y", {}, r"\input{secrets}")


def test_tei_source_only_extracts_article_body():
    xml = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><head>Methods</head><p>The model is trained using a calibrated preference objective.</p></div></body><back><p>Reference text should not enter the evidence index.</p></back></text></TEI>'
    items = parse_tei(xml)
    assert len(items) == 1
    assert items[0]["section"] == "Methods"
    with pytest.raises(ValueError):
        parse_tei(b"<!DOCTYPE x><TEI/>")


def test_visual_questions_attach_actual_pdf_pixels(sample_pdf):
    from types import SimpleNamespace

    paper = SimpleNamespace(metadata=SimpleNamespace(source_pdf=str(sample_pdf)))
    evidence = [ChatEvidence(id="FIG:1", source="paper", kind="figure", text="Plot", page=2)]
    messages, pages = visual_messages(paper, evidence, "Read the axes")
    assert pages == ["2"]
    assert any(block.get("type") == "image_url" for block in messages[0]["content"])


def test_numbered_equations_keep_separate_regions():
    import fitz

    from paper_agent.math_regions import extract_math_regions

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((200, 100), "x = y + 2", fontname="cour", fontsize=12)
        page.insert_text((200, 120), "z = x + 3", fontname="cour", fontsize=12)
        page.insert_text((520, 100), "(1)", fontsize=12)
        page.insert_text((520, 120), "(2)", fontsize=12)
        regions = extract_math_regions(page.get_text("dict"), 1, page.rect.width, page.rect.height)
    numbered = [r for r in regions if r.get("equation_label")]
    assert {r["equation_label"] for r in numbered} == {"1", "2"}
    assert all(not ("x = y" in r["raw"] and "z = x" in r["raw"]) for r in numbered)


def test_correction_recovers_a_previously_blocked_equation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from paper_agent.math_agent import explain_equation
    from tests.test_math_agent import judge_result, math_payload, parsed_math_paper

    monkeypatch.setenv("PAPER_READER_MATH_VISION_ENABLED", "false")
    workspace = Workspace(tmp_path / "work")
    paper = parsed_math_paper()
    layout = {"geometry_latex": r"z = \frac{x}{y}"}
    blocked = explain_equation(
        paper, workspace, 1, "correction-test", "s = x^2 + y", layout_evidence=layout
    )
    assert blocked.status == "transcription_review_required"
    save_correction(workspace, 1, "s = x^2 + y", layout, "s = x^2 + y")

    class Agent:
        def invoke(self, prompt):
            import json

            return {
                "messages": [
                    SimpleNamespace(
                        content=json.dumps(math_payload("Square the input and add the target."))
                    )
                ]
            }

    recovered = explain_equation(
        paper,
        workspace,
        1,
        "correction-test",
        "s = x^2 + y",
        layout_evidence=layout,
        chat_model=object(),
        agent_factory=lambda **kwargs: Agent(),
        judge_fn=lambda *args, **kwargs: judge_result("pass", 4.5),
    )
    assert recovered.latex_source == "user_correction"
    assert recovered.status == "agent_explained"
    assert recovered.display_latex == "s = x^2 + y"

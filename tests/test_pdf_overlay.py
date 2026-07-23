from pathlib import Path

from paper_agent.parser import parse_paper
from paper_agent.term_highlighter import SignificantTerm
from ui.paper_reader_server import page_layout_payload


def test_pdf_layout_uses_original_page_coordinates(tmp_path):
    pdf = Path(__file__).resolve().parents[1] / "examples" / "2606.09079v2.pdf"
    parsed = parse_paper(pdf, tmp_path / "report")
    term = SignificantTerm(
        term="model",
        category="method",
        score=20,
        paper_occurrences=1,
        method_occurrences=1,
        evidence=[],
        skills=["paper-term-highlighter"],
    )

    layout = page_layout_payload(parsed, [term], 1)

    assert layout["width"] > 0
    assert layout["height"] > 0
    assert layout["words"]
    assert any(mark["term"] == "model" for mark in layout["highlights"])


def test_pdf_layout_exposes_hoverable_math_regions(tmp_path):
    pdf = Path(__file__).resolve().parents[1] / "examples" / "2606.09079v2.pdf"
    parsed = parse_paper(pdf, tmp_path / "report")

    layout = page_layout_payload(parsed, [], 3)

    assert layout["mathRegions"]
    assert all(region["raw"] for region in layout["mathRegions"])
    display = next(
        region
        for region in layout["mathRegions"]
        if region["kind"] == "display" and "IUQ" in region["raw"]
    )
    assert display["layout_evidence"]["geometry_latex"] == (
        r"\begin{aligned} c_{t}^{Q} &= h_{t} \cdot W^{DQ}, \\ "
        r"[q_{t,1}^{l}; q_{t,2}^{l}; \ldots; q_{t,n_{h}^{l}}^{l}] "
        r"&= q_{t}^{l} = c_{t}^{Q} \cdot W^{IUQ}, \end{aligned}"
    )
    assert any(
        region["kind"] == "display" and len(region["rects"]) >= 2
        for region in layout["mathRegions"]
    )
    assert any(
        region["kind"] == "inline"
        and ("where" in region["raw"].lower() or "where" in region["context"].lower())
        for region in layout["mathRegions"]
    )


def test_pdf_private_use_glyphs_never_become_display_latex(tmp_path):
    pdf = Path(__file__).resolve().parents[1] / "examples" / "2606.09079v2.pdf"
    parsed = parse_paper(pdf, tmp_path / "report")

    layout = page_layout_payload(parsed, [], 4)
    score = next(region for region in layout["mathRegions"] if "ReLU" in region["raw"])

    assert score["layout_evidence"]["geometry_latex"] == ""

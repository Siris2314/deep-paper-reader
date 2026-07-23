from paper_agent.math_regions import _geometry_latex
from paper_agent.parser import parse_paper
from paper_agent.term_highlighter import SignificantTerm
from ui.paper_reader_server import page_layout_payload


def test_pdf_layout_uses_original_page_coordinates(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
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


def test_pdf_layout_exposes_hoverable_math_regions(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")

    layout = page_layout_payload(parsed, [], 1)

    assert layout["mathRegions"]
    assert all(region["raw"] for region in layout["mathRegions"])
    display = next(region for region in layout["mathRegions"] if region["kind"] == "display")
    assert display["rects"]
    assert "where" in display["context"].lower()


def test_pdf_private_use_glyphs_never_become_display_latex():
    spans = [
        {
            "text": "q = x",
            "x": 10,
            "top": 20,
            "bottom": 32,
            "size": 12,
            "source_block": 0,
        },
        {
            "text": "\ue000",
            "x": 38,
            "top": 12,
            "bottom": 18,
            "size": 6,
            "source_block": 0,
        },
    ]

    assert _geometry_latex(spans) == ""

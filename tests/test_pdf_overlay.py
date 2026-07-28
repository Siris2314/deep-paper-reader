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


def test_pdf_geometry_reconstructs_summation_limits_across_blocks():
    spans = [
        {"text": "L", "x": 303, "top": 274, "bottom": 281, "size": 7, "font": "Math7"},
        {
            "text": "X",
            "x": 299,
            "top": 282,
            "bottom": 292,
            "size": 10,
            "font": "LMMathExtension10-Regular",
        },
        {"text": "V", "x": 270, "top": 285, "bottom": 295, "size": 10, "font": "Math10"},
        {"text": "i,s", "x": 276, "top": 289, "bottom": 296, "size": 7, "font": "Math7"},
        {"text": "=", "x": 288, "top": 285, "bottom": 295, "size": 10, "font": "Roman10"},
        {"text": "l", "x": 300, "top": 299, "bottom": 306, "size": 7, "font": "Math7"},
        {"text": "=1", "x": 302, "top": 299, "bottom": 306, "size": 7, "font": "Roman7"},
        {"text": "I", "x": 315, "top": 283, "bottom": 293, "size": 10, "font": "MSBM10"},
        {"text": "(", "x": 319, "top": 285, "bottom": 295, "size": 10, "font": "Roman10"},
        {"text": "s", "x": 323, "top": 285, "bottom": 295, "size": 10, "font": "Math10"},
        {"text": "∈M", "x": 327, "top": 285, "bottom": 295, "size": 10, "font": "Math10"},
        {"text": "i,l", "x": 352, "top": 289, "bottom": 296, "size": 7, "font": "Math7"},
        {"text": ")", "x": 360, "top": 285, "bottom": 295, "size": 10, "font": "Roman10"},
    ]

    assert _geometry_latex(spans) == (
        r"V_{i,s} = \sum_{l=1}^{L} \mathbb{I} (s \in M_{i,l})"
    )


def test_pdf_geometry_reconstructs_union_limits_without_attaching_them_to_target():
    spans = [
        {"text": "t", "x": 290, "top": 415, "bottom": 422, "size": 7, "font": "Math7"},
        {"text": "+", "x": 293, "top": 415, "bottom": 422, "size": 7, "font": "Roman7"},
        {"text": "τ", "x": 299, "top": 415, "bottom": 422, "size": 7, "font": "Math7"},
        {"text": "−", "x": 304, "top": 415, "bottom": 422, "size": 7, "font": "Math7"},
        {"text": "1", "x": 310, "top": 415, "bottom": 422, "size": 7, "font": "Roman7"},
        {
            "text": "[",
            "x": 296,
            "top": 423,
            "bottom": 433,
            "size": 10,
            "font": "LMMathExtension10-Regular",
        },
        {"text": "Y", "x": 263, "top": 425, "bottom": 435, "size": 10, "font": "Math10"},
        {"text": "+", "x": 270, "top": 423, "bottom": 430, "size": 7, "font": "Roman7"},
        {"text": "t", "x": 269, "top": 430, "bottom": 437, "size": 7, "font": "Math7"},
        {"text": "=", "x": 280, "top": 425, "bottom": 435, "size": 10, "font": "Roman10"},
        {"text": "i", "x": 296, "top": 439, "bottom": 446, "size": 7, "font": "Math7"},
        {"text": "=", "x": 299, "top": 439, "bottom": 446, "size": 7, "font": "Roman7"},
        {"text": "t", "x": 305, "top": 439, "bottom": 446, "size": 7, "font": "Math7"},
        {"text": "A", "x": 316, "top": 425, "bottom": 435, "size": 10, "font": "Math10"},
        {
            "text": "golden",
            "x": 323,
            "top": 423,
            "bottom": 430,
            "size": 7,
            "font": "Roman7",
        },
        {"text": "i", "x": 323, "top": 430, "bottom": 437, "size": 7, "font": "Math7"},
    ]

    assert _geometry_latex(spans) == (
        r"Y_{t}^{+} = \bigcup_{i=t}^{t+\tau-1} A_{i}^{\mathrm{golden}}"
    )

from pathlib import Path

from paper_agent.agents import _lean_paper_bundle
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata


def test_full_report_initial_bundle_is_retrieval_first_and_bounded():
    abstract = "Abstract contribution. " * 600
    method = "Method detail that must be retrieved on demand. " * 4000
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="A Long Paper",
            authors_guess=[],
            page_count=20,
            source_pdf=str(Path("paper.pdf")),
        ),
        full_text=abstract + method,
        page_text={1: abstract, 2: method},
        sections={"abstract": abstract, "method": method},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )

    bundle = _lean_paper_bundle(parsed)

    assert len(bundle) <= 12_000
    assert "Detected sections: abstract, method" in bundle
    assert "retrieve" in bundle.lower()
    assert "Method detail that must be retrieved on demand." not in bundle

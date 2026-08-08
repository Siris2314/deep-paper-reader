import fitz

from paper_agent.parser import parse_paper
from paper_agent.term_highlighter import extract_significant_terms


def test_significant_terms_include_ml_or_method_terms(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    terms = extract_significant_terms(parsed, max_terms=40)

    assert terms
    assert any(term.category in {"math", "method", "result", "concept"} for term in terms)
    assert all("paper-term-highlighter" in term.skills for term in terms)


def test_user_taught_term_outranks_heuristic_terms(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    learned = {
        "model": {
            "term": "model",
            "category": "method",
            "correction_count": 2,
        }
    }

    terms = extract_significant_terms(parsed, max_terms=100, learned_terms=learned)
    model = next(term for term in terms if term.term.lower() == "model")

    assert model.learned is True
    assert model.correction_count == 2
    assert model.category == "method"


def test_phrase_first_extraction_finds_research_spans_and_ignores_metadata(tmp_path):
    pdf = tmp_path / "research-phrases.pdf"
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((18, 500), "arXiv:2305.18290v3", fontsize=9, rotate=90)
    page.insert_text((54, 80), "Abstract", fontsize=14)
    page.insert_text(
        (54, 110),
        "Large unsupervised language models learn broad capabilities from training data.",
        fontsize=10,
    )
    page.insert_text(
        (54, 130),
        "We fit a reward model and then use reinforcement learning from human feedback.",
        fontsize=10,
    )
    document.save(pdf)
    document.close()

    parsed = parse_paper(pdf, tmp_path / "report")
    terms = extract_significant_terms(parsed, max_terms=100, learned_terms={}, vocabulary=[])
    names = {term.term.casefold() for term in terms}

    assert "large unsupervised language models" in names
    assert "reward model" in names
    assert "reinforcement learning" in names
    assert "arxiv" not in names
    assert "model" not in names
    assert "training" not in names

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

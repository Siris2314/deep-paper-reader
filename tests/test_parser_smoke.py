from paper_agent.parser import parse_paper


def test_parser_smoke(tmp_path, sample_pdf):
    parsed = parse_paper(sample_pdf, tmp_path / "report")
    assert parsed.metadata.page_count >= 1
    assert parsed.full_text
    assert (tmp_path / "report" / "parsed" / "metadata.json").exists()
    assert (tmp_path / "report" / "parsed" / "sections.json").exists()

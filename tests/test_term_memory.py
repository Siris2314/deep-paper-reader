from paper_agent.term_memory import load_learned_terms, remember_term, term_memory_prompt_path


def test_remember_term_persists_and_updates(tmp_path):
    path = tmp_path / "agent_memory" / "term_highlights.json"

    first = remember_term(
        "Lookahead Sparse Attention",
        "method",
        paper_title="Example Paper",
        source_pdf="example.pdf",
        page=1,
        path=path,
    )
    second = remember_term(
        "Lookahead Sparse Attention",
        "method",
        paper_title="Another Paper",
        source_pdf="another.pdf",
        page=2,
        path=path,
    )

    learned = load_learned_terms(path)
    assert first["correction_count"] == 1
    assert second["correction_count"] == 2
    assert learned["lookahead sparse attention"]["category"] == "method"
    assert len(learned["lookahead sparse attention"]["examples"]) == 2
    assert "Lookahead Sparse Attention [method]" in term_memory_prompt_path(path).read_text(
        encoding="utf-8"
    )

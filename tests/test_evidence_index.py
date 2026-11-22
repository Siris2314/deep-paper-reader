from paper_agent.evidence_index import (
    build_evidence_records,
    render_source_crop,
    resolve_query,
    search_records,
)
from paper_agent.parser import ParsedPaper, extract_pdf_text
from paper_agent.schemas import TableCard
from paper_agent.workspace import Workspace
from tests.test_paper_chat import sample_paper


def test_pdf_evidence_preserves_source_coordinates(sample_pdf, tmp_path):
    full, pages, metadata = extract_pdf_text(sample_pdf)
    paper = ParsedPaper(metadata, full, pages, {}, [], [], [])
    records = build_evidence_records(paper, Workspace(tmp_path / "work"))
    hits, diagnostics = search_records(records, "retrieval accuracy half memory", 5)
    assert hits[0]["page"] == 2
    assert hits[0]["bbox"] and hits[0]["page_width"] == 612
    png = render_source_crop(paper, 2, hits[0]["bbox"])
    assert png.startswith(b"\x89PNG")
    assert diagnostics["mode"] == "bm25"


def test_late_artifact_is_retrieved_by_number():
    paper = sample_paper()
    paper.table_cards = [
        TableCard(id=f"table_{i}", page=3, caption=f"Table {i}: results", raw_text=f"result {i}")
        for i in range(1, 18)
    ]
    hits, _ = search_records(build_evidence_records(paper), "Explain Table 17", 3)
    assert hits[0]["id"] == "TABLE:table_17"


def test_followup_retains_prior_topic():
    query = resolve_query(
        "Why does that cancel?", [{"role": "user", "content": "Explain the partition function."}]
    )
    assert "partition function" in query


def test_embedding_cache_and_semantic_only_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_RETRIEVAL_EMBEDDING_PROVIDER", "huggingface")
    calls = []

    def embed(texts, model, *, query=False):
        calls.append(query)
        return [[0, 1] if query or "rejected" in text else [1, 0] for text in texts]

    monkeypatch.setattr("paper_agent.evidence_index._embed", embed)
    records = [
        {"id": "a", "text": "rejected responses", "kind": "page"},
        {"id": "b", "text": "hardware", "kind": "page"},
    ]
    workspace = Workspace(tmp_path / "work")
    first, info = search_records(records, "dispreferred completions", 1, workspace)
    second, info = search_records(records, "dispreferred completions", 1, workspace)
    assert first[0]["id"] == second[0]["id"] == "a"
    assert calls == [False, True, True]
    assert info["embedding_cache_hit"] is True
    assert info["mode"] == "hybrid_rrf"


def test_no_match_abstains():
    hits, _ = search_records(
        [{"id": "a", "text": "sparse attention", "kind": "page"}], "monoclonal antibody dose"
    )
    assert hits == []

from paper_agent.parser import ParsedPaper
from paper_agent.research_lineage import (
    build_research_lineage,
    citation_contexts,
    parse_bibliography,
    semantic_scholar_references,
)
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace


def lineage_paper() -> ParsedPaper:
    body = (
        "2 Method\nFollowing the retrieval architecture of Prior Memory [1], we build upon it by "
        "replacing dense lookup with sparse learned indexing. We evaluate long-context retrieval on "
        "the RULER benchmark [2]. A general survey provides background [3]."
    )
    references = (
        "References\n"
        "[1] A. Author. Prior Memory: Retrieval for Long Context. 2023. https://example.org/prior\n"
        "[2] B. Author. RULER: A Long Context Benchmark. 2024. https://example.org/ruler\n"
        "[3] C. Author. A Survey of Language Models. 2022. https://example.org/survey"
    )
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Sparse Memory", authors_guess=[], page_count=2, source_pdf="2601.12345.pdf"
        ),
        full_text=f"{body}\n{references}",
        page_text={1: body, 2: references},
        sections={"method": body, "references": references},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_parses_bibliography_and_exact_citation_pages():
    parsed = lineage_paper()

    entries = parse_bibliography(parsed)
    contexts = citation_contexts(parsed)

    assert [item.number for item in entries] == [1, 2, 3]
    assert entries[0].title.startswith("Prior Memory")
    assert contexts[1][0].page == 1
    assert "build upon" in contexts[1][0].text


def test_lineage_classifies_explicit_relationships_conservatively(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED", raising=False)
    workspace = Workspace(tmp_path / "workspace")

    lineage = build_research_lineage(lineage_paper(), workspace, "retrieval")
    relations = {item.reference_number: item for item in lineage.relations}

    assert relations[1].relation == "extends"
    assert relations[1].confidence == "high"
    assert relations[2].relation == "evaluates_on"
    assert relations[3].relation == "background"
    assert lineage.external_status == "not_configured"


def test_lineage_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    workspace = Workspace(tmp_path / "workspace")

    first = build_research_lineage(lineage_paper(), workspace, "memory")
    second = build_research_lineage(lineage_paper(), workspace, "memory")

    assert second.summary == first.summary
    assert second.relations[0].contexts[0].page == 1


def test_reference_continuation_pages_are_not_citation_contexts():
    parsed = lineage_paper()
    parsed.page_text = {
        1: parsed.page_text[1],
        2: "References\n[1] A. Author. Prior Memory. 2023.",
        3: "[2] B. Author. RULER. 2024.",
    }

    contexts = citation_contexts(parsed)

    assert all(item.page == 1 for items in contexts.values() for item in items)


def test_wrapped_reference_url_keeps_filename():
    parsed = lineage_paper()
    parsed.sections["references"] = (
        "References\n[1] A. Author. Prior Memory. 2023. https://example.org/files/ prior_memory.pdf"
    )

    entries = parse_bibliography(parsed)

    assert entries[0].url == "https://example.org/files/prior_memory.pdf"


def test_external_identity_does_not_trust_arxiv_filename(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setattr(
        "paper_agent.research_lineage._semantic_scholar_json",
        lambda _url: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    status, records = semantic_scholar_references(lineage_paper())

    assert status == "paper_id_unresolved"
    assert records == []


def test_external_identity_uses_first_page_doi(monkeypatch):
    parsed = lineage_paper()
    parsed.page_text[1] = "doi:10.1234/sparse-memory\n" + parsed.page_text[1]
    monkeypatch.setenv("SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED", "true")
    urls = []
    monkeypatch.setattr(
        "paper_agent.research_lineage._semantic_scholar_json",
        lambda url: urls.append(url) or {"data": []},
    )

    status, records = semantic_scholar_references(parsed)

    assert status == "ok"
    assert records == []
    assert "/paper/DOI:10.1234%2Fsparse-memory/references" in urls[0]

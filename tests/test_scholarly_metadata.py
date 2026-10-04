from __future__ import annotations

import io
import json
import urllib.error

import pytest

from paper_agent.parser import ParsedPaper
from paper_agent.parser import parse_paper
from paper_agent.schemas import PaperMetadata
from paper_agent.scholarly_metadata import (
    _openalex,
    scholarly_json_get,
    load_metadata,
    paper_identity,
    refresh_metadata,
    retrieve_metadata,
    save_local_identity,
)
from paper_agent.workspace import Workspace


def paper(first_page: str, *, title: str = "Calibrated Retrieval Networks") -> ParsedPaper:
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess=title, authors_guess=["A. Author"], page_count=2, source_pdf="paper.pdf"
        ),
        full_text=first_page + "\nReferences\n10.9999/reference-only",
        page_text={1: first_page, 2: "References\n10.9999/reference-only"},
        sections={},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


def test_identity_uses_exact_first_page_identifiers_only():
    identity = paper_identity(
        paper("arXiv:2402.08954v1\nCalibrated Retrieval Networks\ndoi:10.1234/Test.42")
    )

    assert identity["doi"] == "10.1234/test.42"
    assert identity["arxiv_version"] == "2402.08954v1"
    assert identity["arxiv_id"] == "2402.08954"
    assert "10.9999" not in str(identity)


def test_ambiguous_first_page_dois_are_not_guessed():
    identity = paper_identity(paper("10.1234/one and 10.5678/two"))

    assert identity["doi"] is None
    assert identity["doi_candidates"] == ["10.1234/one", "10.5678/two"]
    assert identity["status"] == "local_only"


def test_safe_json_get_rejects_non_allowlisted_url():
    with pytest.raises(ValueError, match="allowlisted"):
        scholarly_json_get("https://example.org/metadata")


def test_safe_json_get_retries_once_and_bounds_response(monkeypatch):
    calls = 0

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def geturl(self):
            return "https://api.crossref.org/works/10.1/test"

        def read(self, _):
            return b'{"ok": true}'

    def urlopen(*_, **__):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                "https://api.crossref.org", 503, "busy", {"Retry-After": "0"}, io.BytesIO()
            )
        return Response()

    monkeypatch.setattr("paper_agent.scholarly_metadata.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("paper_agent.scholarly_metadata.time.sleep", lambda _: None)

    assert scholarly_json_get("https://api.crossref.org/works/10.1/test") == {"ok": True}
    assert calls == 2


def test_openalex_fetches_related_work_only_after_identity_match(monkeypatch):
    urls: list[str] = []

    def fetch(url, _headers):
        urls.append(url)
        if "filter=" in url:
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W2",
                        "display_name": "A Related Method",
                        "publication_year": 2024,
                        "cited_by_count": 7,
                        "open_access": {"is_oa": True},
                    }
                ]
            }
        return {
            "id": "https://openalex.org/W1",
            "display_name": "Calibrated Retrieval Networks",
            "related_works": ["https://openalex.org/W2"],
            "authorships": [],
        }

    monkeypatch.setattr("paper_agent.scholarly_metadata.scholarly_json_get", fetch)
    result = _openalex(paper_identity(paper("doi:10.1234/test")))

    assert result["status"] == "ready"
    assert result["related_works"][0]["title"] == "A Related Method"
    assert len(urls) == 2
    assert "select=" in urls[0]


def test_openalex_title_mismatch_does_not_expand_related_graph(monkeypatch):
    calls = 0

    def fetch(_url, _headers):
        nonlocal calls
        calls += 1
        return {
            "id": "https://openalex.org/W1",
            "display_name": "A Completely Unrelated Clinical Trial",
            "related_works": ["https://openalex.org/W2"],
            "authorships": [],
        }

    monkeypatch.setattr("paper_agent.scholarly_metadata.scholarly_json_get", fetch)
    result = _openalex(paper_identity(paper("doi:10.1234/test")))

    assert result["status"] == "identity_mismatch"
    assert result["related_works"] == []
    assert calls == 1


def test_refresh_is_document_bound_and_negative_cached(tmp_path, monkeypatch):
    parsed = paper("doi:10.1234/test")
    workspace = Workspace(tmp_path)
    save_local_identity(parsed, workspace)
    calls = 0

    def unavailable(_identity):
        nonlocal calls
        calls += 1
        raise OSError("offline")

    monkeypatch.setattr("paper_agent.scholarly_metadata._crossref", unavailable)
    monkeypatch.setattr("paper_agent.scholarly_metadata._openalex", unavailable)
    monkeypatch.setattr("paper_agent.scholarly_metadata._semantic_scholar", unavailable)

    assert refresh_metadata(parsed, workspace)["status"] == "unavailable"
    assert refresh_metadata(parsed, workspace)["status"] == "unavailable"
    assert calls == 3
    changed = paper("doi:10.5678/different")
    assert load_metadata(changed, workspace)["status"] == "not_loaded"


def test_cached_metadata_is_retrieved_only_for_discovery_questions(tmp_path):
    parsed = paper("doi:10.1234/test")
    workspace = Workspace(tmp_path)
    payload = save_local_identity(parsed, workspace)
    payload.update(
        status="ready",
        providers={
            "openalex": {
                "status": "ready",
                "id": "https://openalex.org/W1",
                "title": "Calibrated Retrieval Networks",
                "related_works": [{"id": "https://openalex.org/W2", "title": "A Related Method"}],
            }
        },
    )
    workspace.write_json("research/scholarly_metadata.json", payload)

    assert retrieve_metadata(parsed, workspace, "Explain the loss function") == []
    results = retrieve_metadata(parsed, workspace, "What related work is similar?")
    assert [item["id"] for item in results] == ["META:openalex", "RELATED:1"]
    assert json.loads(results[1]["text"])["title"] == "A Related Method"


def test_parse_persists_document_bound_local_identity(tmp_path, sample_pdf):
    workspace = tmp_path / "workspace"
    parsed = parse_paper(sample_pdf, workspace)

    cached = load_metadata(parsed, Workspace(workspace))

    assert cached["document_key"]
    assert cached["identity"]["evidence"].startswith("Identifiers extracted")

from types import SimpleNamespace

from paper_agent.concept_enrichment import (
    TavilyTermResearch,
    _tavily_http,
    enrich_concept_card,
    run_tavily_term_research,
    synthesize_why_it_matters,
)
from paper_agent.parser import ParsedPaper
from paper_agent.schemas import PaperMetadata
from paper_agent.workspace import Workspace


def parsed_kv_cache_paper() -> ParsedPaper:
    evidence = (
        "Conventional LLMs keep the full KV cache loaded during decoding, causing a severe GPU memory bottleneck. "
        "The proposed indexer predicts useful chunks so the system can reduce the physical KV cache footprint."
    )
    return ParsedPaper(
        metadata=PaperMetadata(
            title_guess="A KV Cache Paper",
            authors_guess=[],
            page_count=1,
            source_pdf="paper.pdf",
        ),
        full_text=evidence,
        page_text={1: evidence},
        sections={"method": evidence},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )


class FakeTool:
    def __init__(self, response):
        self.response = response

    def invoke(self, payload):
        return self.response


def test_tavily_http_has_a_bounded_request_timeout(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"status": "pending", "request_id": "request-1"}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return Response()

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_HTTP_TIMEOUT", "17")
    monkeypatch.setattr("paper_agent.concept_enrichment.requests.request", fake_request)

    _tavily_http("POST", "research", {"input": "define KV cache"})

    assert captured["timeout"] == (5.0, 17.0)


def test_tavily_research_content_becomes_general_explanation():
    research = run_tavily_term_research(
        "KV cache",
        "A KV Cache Paper",
        research_tool=FakeTool(
            {
                "status": "completed",
                "request_id": "request-1",
                "model": "mini",
                "content": {
                    "general_explanation": "A KV cache stores transformer attention keys and values."
                },
                "sources": [{"title": "Transformer caching", "url": "https://example.com/cache"}],
            }
        ),
    )

    assert research.general_explanation.startswith("A KV cache")
    assert research.source_type == "tavily_research"
    assert research.model == "mini"
    assert research.sources[0]["url"] == "https://example.com/cache"


def test_search_fallback_uses_a_query_below_tavily_limit():
    captured = {}

    class SearchTool:
        def invoke(self, payload):
            captured.update(payload)
            return {
                "answer": "A KV cache stores reusable attention keys and values.",
                "results": [],
            }

    research = run_tavily_term_research(
        "KV cache",
        "A Paper Title " * 80,
        research_tool=FakeTool({"error": "Research unavailable"}),
        search_tool=SearchTool(),
    )

    assert len(captured["query"]) <= 400
    assert "KV cache" in captured["query"]
    assert research.source_type == "tavily_search_answer"


def test_why_it_matters_comes_from_paper_agent(tmp_path):
    class FakeAgent:
        def invoke(self, payload):
            assert "GPU memory bottleneck" in payload["messages"][0]["content"]
            return {
                "messages": [
                    SimpleNamespace(
                        content="KV-cache pressure is the paper's central serving bottleneck. Reducing it enables the reported long-context system design."
                    )
                ]
            }

    answer, model = synthesize_why_it_matters(
        parsed_kv_cache_paper(),
        Workspace(tmp_path / "workspace"),
        "KV cache",
        chat_model=object(),
        agent_factory=lambda **kwargs: FakeAgent(),
    )

    assert "central serving bottleneck" in answer
    assert ":" in model


def test_enriched_card_records_field_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "paper_agent.concept_enrichment.run_tavily_term_research",
        lambda term, title: TavilyTermResearch(
            general_explanation="A researched definition.",
            sources=[{"title": "Source", "url": "https://example.com", "snippet": ""}],
            source_type="tavily_research",
            model="mini",
            request_id="request-2",
        ),
    )
    monkeypatch.setattr(
        "paper_agent.concept_enrichment.synthesize_why_it_matters",
        lambda parsed, workspace, term: (
            "A paper-grounded relevance explanation.",
            "ollama:test-model",
        ),
    )
    workspace = Workspace(tmp_path / "workspace")

    card = enrich_concept_card(parsed_kv_cache_paper(), workspace, "KV cache")

    assert card.general_explanation_source == "tavily_research"
    assert card.why_it_matters_source == "paper_agent"
    assert card.status == "enriched"


def test_retry_reuses_cached_tavily_research(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    workspace.write_json(
        "web/tavily_term_research/kv-cache.json",
        {
            "general_explanation": "Cached Tavily definition.",
            "sources": [],
            "source_type": "tavily_research",
            "model": "mini",
            "request_id": "cached-request",
        },
    )
    monkeypatch.setattr(
        "paper_agent.concept_enrichment.run_tavily_term_research",
        lambda term, title: (_ for _ in ()).throw(AssertionError("Tavily should not run again")),
    )
    monkeypatch.setattr(
        "paper_agent.concept_enrichment.synthesize_why_it_matters",
        lambda parsed, workspace, term: ("Agent retry succeeded.", "ollama:qwen2.5:7b"),
    )

    card = enrich_concept_card(parsed_kv_cache_paper(), workspace, "KV cache")

    assert card.general_explanation == "Cached Tavily definition."
    assert card.why_it_matters_here == "Agent retry succeeded."


def test_paper_agent_falls_back_to_next_local_model(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, name):
            self.name = name

        def model_copy(self, update):
            return self

    class FakeAgent:
        def __init__(self, model):
            self.model = model

        def invoke(self, payload):
            if self.model.name == "slow-model":
                raise TimeoutError("timed out")
            return {"messages": [SimpleNamespace(content="Fallback model completed the analysis.")]}

    monkeypatch.setenv("PAPER_READER_AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("PAPER_READER_AGENT_MODEL", "slow-model")
    monkeypatch.setenv("PAPER_READER_AGENT_FALLBACK_MODELS", "fast-model")
    monkeypatch.setattr(
        "paper_agent.concept_enrichment.build_chat_model",
        lambda config: FakeModel(config.model),
    )

    answer, model = synthesize_why_it_matters(
        parsed_kv_cache_paper(),
        Workspace(tmp_path / "workspace"),
        "KV cache",
        agent_factory=lambda **kwargs: FakeAgent(kwargs["model"]),
    )

    assert "Fallback model" in answer
    assert model == "ollama:fast-model"

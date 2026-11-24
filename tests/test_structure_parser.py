import json

from paper_agent.parser import load_parsed_paper_from_workspace, parse_paper
from paper_agent.structure_parser import _batches, parse_structure
from paper_agent.workspace import Workspace


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def with_structured_output(self, schema, method):
        assert method == "json_schema"
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


def test_model_structure_is_anchored_to_source_lines(monkeypatch):
    monkeypatch.setenv("PAPER_STRUCTURE_ENABLED", "true")
    pages = {
        1: "A New Method\nAda Researcher\nAbstract\nA method for retrieval.\n1 Introduction\nWe explain it.\nFigure 1: Retrieval quality\nq = softmax(x)",
        2: "2 Results\nAccuracy increased.\nTable 1: Benchmarks\nModel 92 percent",
    }
    model = FakeModel(
        {
            "pages": [
                {
                    "page": 1,
                    "title_lines": [0],
                    "author_lines": [1],
                    "headings": [
                        {"line": 2, "kind": "abstract"},
                        {"line": 4, "kind": "introduction"},
                    ],
                    "equations": [{"start": 7, "end": 7}, {"start": 100, "end": 100}],
                    "figures": [{"start": 6, "end": 6}],
                },
                {
                    "page": 2,
                    "headings": [{"line": 0, "kind": "results"}],
                    "tables": [{"start": 2, "end": 2}],
                },
            ]
        }
    )

    result = parse_structure(pages, model)

    assert result.status["state"] == "ok"
    assert result.title == "A New Method"
    assert result.authors == ["Ada Researcher"]
    assert result.sections["abstract"] == "A method for retrieval."
    assert result.sections["introduction"].startswith("We explain it.")
    assert result.sections["results"].startswith("Accuracy increased.")
    assert [card.raw for card in result.equations] == ["q = softmax(x)"]
    assert result.figures[0].caption == "Figure 1: Retrieval quality"
    assert result.tables[0].caption == "Table 1: Benchmarks"
    assert result.markers == [
        {"label": "Abstract", "page": 1},
        {"label": "1 Introduction", "page": 1},
        {"label": "2 Results", "page": 2},
    ]
    assert model.calls


def test_model_failure_keeps_raw_pages_without_regex_fallback(monkeypatch):
    monkeypatch.setenv("PAPER_STRUCTURE_ENABLED", "true")

    class Unavailable(FakeModel):
        def invoke(self, messages):
            raise RuntimeError("Ollama unavailable")

    result = parse_structure({1: "1 Introduction\nA source paragraph."}, Unavailable({}))
    assert result.status["state"] == "unavailable"
    assert result.sections == {"front_matter": "1 Introduction\nA source paragraph."}
    assert not result.markers
    assert not result.equations


def test_parse_persists_model_structure(tmp_path, sample_pdf, monkeypatch):
    monkeypatch.setenv("PAPER_STRUCTURE_ENABLED", "true")
    model = FakeModel(
        {
            "pages": [
                {"page": 1, "title_lines": [0], "headings": [{"line": 2, "kind": "abstract"}]},
                {"page": 2, "headings": [{"line": 0, "kind": "method"}]},
            ]
        }
    )
    workspace = Workspace(tmp_path / "report")
    parsed = parse_paper(sample_pdf, workspace.root, structure_model=model)
    reloaded = load_parsed_paper_from_workspace(workspace)

    assert parsed.metadata.title_guess == "Sparse Attention for Efficient Language Models"
    assert reloaded.section_markers == parsed.section_markers
    assert reloaded.sections["abstract"]
    assert json.loads(workspace.path("parsed/structure_status.json").read_text())["state"] == "ok"


def test_batching_preserves_every_line_of_long_pages():
    pages = {1: "\n".join(f"line {index} " + "x" * 80 for index in range(240))}
    batches = _batches(pages)
    seen = [index for batch in batches for _, lines in batch for index, _ in lines]
    assert seen == list(range(240))
    assert len(batches) > 1

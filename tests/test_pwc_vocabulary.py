from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from paper_agent.parser import ParsedPaper
from paper_agent.pwc_vocabulary import VocabularyTerm, load_method_vocabulary
from paper_agent.schemas import PaperMetadata
from paper_agent.term_highlighter import extract_significant_terms


def test_pwc_archive_is_compacted_into_a_local_vocabulary(tmp_path):
    cache_path = tmp_path / "pwc-methods.json"

    def fake_download(_url: str, target: Path, _timeout: float) -> None:
        table = pa.table(
            {
                "name": ["WidgetNet", "WidgetNet"],
                "full_name": ["Widget Retrieval Network", "Widget Retrieval Network"],
                "num_papers": [4, 9],
            }
        )
        pq.write_table(table, target)

    first = load_method_vocabulary(
        cache_path=cache_path,
        allow_network=True,
        downloader=fake_download,
    )
    second = load_method_vocabulary(cache_path=cache_path, allow_network=False)

    assert cache_path.exists()
    assert {item.term for item in first} == {"WidgetNet", "Widget Retrieval Network"}
    assert next(item for item in first if item.term == "WidgetNet").num_papers == 9
    assert second == first


def test_archive_method_becomes_a_highlight_without_hardcoding():
    text = (
        "We introduce Widget Retrieval Network for efficient search. "
        "Widget Retrieval Network is the main method in our evaluation."
    )
    parsed = ParsedPaper(
        metadata=PaperMetadata(
            title_guess="Widget Paper",
            authors_guess=[],
            page_count=1,
            source_pdf="paper.pdf",
        ),
        full_text=text,
        page_text={1: text},
        sections={"method": text},
        equation_cards=[],
        figure_cards=[],
        table_cards=[],
    )
    vocabulary = [VocabularyTerm("Widget Retrieval Network", "Widget Retrieval Network", 7)]

    terms = extract_significant_terms(parsed, learned_terms={}, vocabulary=vocabulary)
    widget = next(item for item in terms if item.term == "Widget Retrieval Network")

    assert widget.category == "method"
    assert widget.vocabulary_source == "papers_with_code_archive"

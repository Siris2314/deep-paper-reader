import os
from pathlib import Path

import fitz
import pytest


# Unit tests must not depend on the availability of the external vocabulary archive.
os.environ.setdefault("PWC_VOCAB_ALLOW_NETWORK", "false")


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a small, redistributable paper fixture for parser and overlay tests."""
    path = tmp_path / "sample-paper.pdf"
    document = fitz.open()

    first = document.new_page(width=612, height=792)
    first.insert_text(
        (54, 64),
        "Sparse Attention for Efficient Language Models",
        fontsize=18,
    )
    first.insert_text((54, 92), "Ada Researcher, Lin Scientist", fontsize=11)
    first.insert_text((54, 126), "Abstract", fontsize=14)
    first.insert_text(
        (54, 150),
        "We introduce a sparse attention model for long-context language modeling.",
        fontsize=11,
    )
    first.insert_text(
        (54, 168),
        "The method reduces memory use while preserving retrieval accuracy.",
        fontsize=11,
    )
    first.insert_text((54, 210), "1 Introduction", fontsize=14)
    first.insert_text(
        (54, 234),
        "Standard attention computes a probability distribution over every token.",
        fontsize=11,
    )
    first.insert_text(
        (120, 286),
        "q = softmax(x)",
        fontsize=13,
        fontname="cour",
    )
    first.insert_text(
        (54, 318),
        "where x is the score vector and q contains normalized attention weights.",
        fontsize=11,
    )

    second = document.new_page(width=612, height=792)
    second.insert_text((54, 64), "2 Method", fontsize=14)
    second.insert_text(
        (54, 92),
        "Our sparse attention method selects the top-k keys for each query.",
        fontsize=11,
    )
    second.insert_text(
        (54, 112),
        "This lowers KV cache memory and improves inference throughput.",
        fontsize=11,
    )
    second.insert_text((54, 156), "3 Results", fontsize=14)
    second.insert_text(
        (54, 184),
        "The model reaches 92 percent retrieval accuracy with half the memory.",
        fontsize=11,
    )

    document.save(path)
    document.close()
    return path

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from paper_agent.concepts import concept_card_to_markdown, get_or_create_concept_card
from paper_agent.config import RunConfig
from paper_agent.paper_chat import run_paper_chat
from paper_agent.parser import load_parsed_paper_from_workspace
from paper_agent.workspace import Workspace


def chat_once(config: RunConfig, question: str) -> str:
    """Run one CLI turn through the same routed workflow used by the reader."""
    workspace = Workspace(config.output_dir)
    parsed = load_parsed_paper_from_workspace(workspace)
    mode = "web" if config.research_enabled else "fast"
    result = run_paper_chat(
        parsed,
        workspace,
        question,
        mode=mode,
        thread_id="cli",
    )
    return result.answer


def make_concept_card(config: RunConfig, term: str) -> dict[str, Any]:
    workspace = Workspace(config.output_dir)
    parsed = load_parsed_paper_from_workspace(workspace)
    card = get_or_create_concept_card(parsed, workspace, term)
    workspace.write_text(f"concepts/cards/{card.slug}.md", concept_card_to_markdown(card))
    return asdict(card)

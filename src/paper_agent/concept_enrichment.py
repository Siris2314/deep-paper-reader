from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from paper_agent.agents import build_chat_model
from paper_agent.concepts import (
    ConceptCard,
    create_deterministic_concept_card,
    sentence_windows,
    slugify,
)
from paper_agent.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, RunConfig
from paper_agent.context_diagnostics import record_context_usage
from paper_agent.observability import invoke_observed, paper_session_id, workflow_trace
from paper_agent.parser import ParsedPaper
from paper_agent.term_highlighter import method_section_text
from paper_agent.workspace import Workspace


@dataclass
class TavilyTermResearch:
    general_explanation: str
    sources: list[dict[str, str]]
    source_type: str
    model: str
    request_id: str | None = None


def _tavily_http(method: str, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")
    base_url = os.getenv("TAVILY_API_BASE_URL", "https://api.tavily.com").rstrip("/")
    timeout = max(5.0, min(float(os.getenv("TAVILY_HTTP_TIMEOUT", "20")), 60.0))
    response = requests.request(
        method,
        f"{base_url}/{endpoint.lstrip('/')}",
        json=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Client-Source": "deep-paper-reader",
        },
        timeout=(5.0, timeout),
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text[:500]}
    if response.status_code not in {200, 202}:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise RuntimeError(f"Tavily HTTP {response.status_code}: {detail or payload}")
    return payload if isinstance(payload, dict) else {}


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"content": value}
        except json.JSONDecodeError:
            return {"content": value}
    return {}


def _sources(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")[:500]
        title = str(item.get("title") or url or "Untitled source")[:200]
        key = (url or title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "title": title,
                "url": url,
                "snippet": str(item.get("content") or item.get("snippet") or "")[:800],
            }
        )
    return output[:8]


def _general_explanation(content: Any) -> str:
    if isinstance(content, dict):
        text = (
            content.get("general_explanation")
            or content.get("definition")
            or content.get("summary")
        )
        return str(text or "").strip()
    if not isinstance(content, str):
        return ""
    text = content.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _general_explanation(parsed)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^#{1,4}\s+[^\n]+\n+", "", text)
    return text[:2400].strip()


def _tavily_search_query(term: str) -> str:
    query = (
        f'Explain the machine-learning concept "{term}": give its definition, core mechanism, purpose, '
        "important mathematical behavior, and systems tradeoffs in a concise technical answer."
    )
    # Tavily Search rejects queries over 400 characters. Leave headroom for API-side handling.
    return query[:380]


def run_tavily_term_research(
    term: str,
    paper_title: str,
    *,
    research_tool: Any | None = None,
    get_research_tool: Any | None = None,
    search_tool: Any | None = None,
) -> TavilyTermResearch:
    if not os.getenv("TAVILY_API_KEY") and research_tool is None:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    model = os.getenv("TAVILY_RESEARCH_MODEL", "mini")
    prompt = (
        f"Define and explain the machine-learning term '{term}' for a technically literate reader. "
        "Explain its mechanism, the problem it solves, and any important mathematical or systems behavior. "
        f"The reader encountered it in the paper '{paper_title}', but keep this response general rather than "
        "making claims about that paper. Return two or three short paragraphs totaling roughly 100 to 180 words. "
        "Wrap every mathematical expression in dollar-sign LaTeX delimiters, for example $QK^T$ or "
        "$X \\in \\mathbb{R}^{L \\times d}$. Do not emit raw Unicode pseudo-LaTeX."
    )
    schema = {
        "properties": {
            "general_explanation": {
                "type": "string",
                "description": "A self-contained technical explanation in short paragraphs with dollar-delimited LaTeX.",
            }
        },
        "required": ["general_explanation"],
    }

    try:
        if research_tool is None:
            created = _tavily_http(
                "POST",
                "research",
                {
                    "input": prompt,
                    "model": model,
                    "output_schema": schema,
                    "stream": False,
                    "citation_format": "numbered",
                },
            )
        else:
            created = _payload(research_tool.invoke({"input": prompt}))
        if created.get("error"):
            raise RuntimeError(str(created["error"]))
        result = created
        request_id = str(created.get("request_id") or "") or None
        status = str(created.get("status") or "").lower()
        if status not in {"completed", "failed"}:
            if not request_id:
                raise RuntimeError("Tavily Research did not return a request ID.")
            timeout = max(10.0, min(float(os.getenv("TAVILY_RESEARCH_TIMEOUT", "90")), 300.0))
            interval = max(1.0, min(float(os.getenv("TAVILY_RESEARCH_POLL_INTERVAL", "2")), 10.0))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                result = (
                    _payload(get_research_tool.invoke({"request_id": request_id}))
                    if get_research_tool is not None
                    else _tavily_http("GET", f"research/{request_id}")
                )
                status = str(result.get("status") or "").lower()
                if status in {"completed", "failed"}:
                    break
                time.sleep(interval)
        if str(result.get("status") or "").lower() == "failed":
            raise RuntimeError(str(result.get("error") or "Tavily Research failed."))
        explanation = _general_explanation(result.get("content"))
        if not explanation:
            raise RuntimeError("Tavily Research completed without a general explanation.")
        return TavilyTermResearch(
            general_explanation=explanation,
            sources=_sources(result.get("sources")),
            source_type="tavily_research",
            model=str(result.get("model") or model),
            request_id=request_id,
        )
    except Exception as research_error:
        # Search answers keep the popup useful when the Research API is unavailable, while
        # provenance makes the fallback explicit instead of pretending it was Research.
        search_query = _tavily_search_query(term)
        searched = (
            _tavily_http(
                "POST",
                "search",
                {
                    "query": search_query,
                    "max_results": 5,
                    "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
                    "include_answer": "advanced",
                    "include_raw_content": False,
                },
            )
            if search_tool is None
            else _payload(search_tool.invoke({"query": search_query}))
        )
        explanation = str(searched.get("answer") or "").strip()
        if not explanation:
            raise RuntimeError(
                f"Tavily Research failed ({research_error}); search returned no answer."
            )
        return TavilyTermResearch(
            general_explanation=explanation,
            sources=_sources(searched.get("results")),
            source_type="tavily_search_answer",
            model="search-answer",
            request_id=str(searched.get("request_id") or "") or None,
        )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    else:
        text = str(content or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def synthesize_why_it_matters(
    parsed: ParsedPaper,
    workspace: Workspace,
    term: str,
    *,
    agent_factory: Any | None = None,
    chat_model: Any | None = None,
) -> tuple[str, str]:
    evidence = sentence_windows(parsed.full_text, term, limit=6)
    method_hits = sentence_windows(method_section_text(parsed), term, limit=4)
    evidence_text = "\n".join(f"- {item}" for item in [*method_hits, *evidence][:8])
    if not evidence_text:
        evidence_text = "- No direct sentence-level evidence was retrieved."

    provider = os.getenv("PAPER_READER_AGENT_PROVIDER", os.getenv("MODEL_PROVIDER", "ollama"))
    primary_model = os.getenv(
        "PAPER_READER_AGENT_MODEL",
        os.getenv("MODEL_NAME", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)),
    )
    if agent_factory is None:
        from deepagents import create_deep_agent

        agent_factory = create_deep_agent

    model_names = [primary_model]
    if chat_model is None and provider == "ollama":
        fallbacks = os.getenv("PAPER_READER_AGENT_FALLBACK_MODELS", "qwen2.5:3b")
        model_names.extend(name.strip() for name in fallbacks.split(",") if name.strip())
    model_names = list(dict.fromkeys(model_names))
    failures: list[str] = []
    output_budget = int(os.getenv("PAPER_READER_AGENT_MAX_TOKENS", "220"))
    system_prompt = (
        "You are a paper-grounded concept relevance agent. Explain why one term matters in this "
        "specific paper using only the supplied paper evidence. Connect it to the method, system "
        "bottleneck, or result. Do not give a generic definition, use web knowledge, or claim "
        "centrality without evidence. Return two to four concise sentences with no heading."
    )
    user_prompt = (
        f"Paper: {parsed.metadata.title_guess or 'Untitled paper'}\n"
        f"Term: {term}\n\nPaper evidence:\n{evidence_text}"
    )

    for model_name in model_names:
        current_model = chat_model
        if current_model is None:
            config = RunConfig(
                pdf_path=Path(parsed.metadata.source_pdf),
                output_dir=workspace.root,
                model_provider=provider,
                model=model_name,
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            )
            current_model = build_chat_model(config)
            if provider == "ollama" and hasattr(current_model, "model_copy"):
                current_model = current_model.model_copy(
                    update={
                        "keep_alive": os.getenv("PAPER_READER_AGENT_KEEP_ALIVE", "15m"),
                        "num_predict": output_budget,
                    }
                )
        try:
            record_context_usage(
                workspace,
                workflow="concept_relevance",
                provider=provider,
                model=model_name,
                components={"system prompt": system_prompt, "paper evidence prompt": user_prompt},
                reserved_output_tokens=output_budget,
                metadata={"term": term},
            )
            agent = agent_factory(
                model=current_model,
                tools=[],
                system_prompt=system_prompt,
                name="concept_relevance_agent",
            )
            result = invoke_observed(
                agent,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ]
                },
                name="concept-relevance-agent",
                model=f"{provider}:{model_name}",
                metadata={"term": term},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            answer = _message_text(messages[-1]) if messages else _message_text(result)
            if not answer:
                raise RuntimeError("empty response")
            return answer, f"{provider}:{model_name}"
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")
        if chat_model is not None:
            break
    raise RuntimeError("; ".join(failures) or "No paper-agent model was available.")


def _enrich_concept_card_impl(
    parsed: ParsedPaper,
    workspace: Workspace,
    term: str,
    progress: Callable[[str, str], None] | None = None,
) -> ConceptCard:
    base = create_deterministic_concept_card(parsed, workspace, term)
    research_rel = f"web/tavily_term_research/{slugify(term)}.json"
    research_path = workspace.path(research_rel)
    research: TavilyTermResearch | None = None
    if research_path.exists():
        try:
            research = TavilyTermResearch(**json.loads(research_path.read_text(encoding="utf-8")))
        except Exception:
            research = None
    if research and research.general_explanation:
        if progress:
            progress("tavily_cached", "Using cached Tavily research; no new web request is needed.")
    else:
        if progress:
            progress("tavily_research", "Tavily is researching the general definition.")
        research = run_tavily_term_research(term, parsed.metadata.title_guess or "research paper")
        research_path = workspace.write_json(research_rel, asdict(research))
    try:
        if progress:
            progress("paper_agent", "The local paper agent is analyzing why the term matters here.")
        why, agent_model = synthesize_why_it_matters(parsed, workspace, term)
        why_source = "paper_agent"
    except Exception as exc:
        why = f"Paper-agent synthesis failed: {exc}"
        agent_model = "unavailable"
        why_source = "paper_agent_error"

    card = ConceptCard(
        term=base.term,
        slug=base.slug,
        paper_specific_meaning=base.paper_specific_meaning,
        general_explanation=research.general_explanation,
        why_it_matters_here=why,
        prerequisites=base.prerequisites,
        paper_evidence=base.paper_evidence,
        web_sources=research.sources,
        status="enriched" if why_source == "paper_agent" else "partially_enriched",
        general_explanation_source=research.source_type,
        general_explanation_model=research.model,
        why_it_matters_source=why_source,
        why_it_matters_model=agent_model,
    )
    workspace.write_json(f"concepts/cards/{card.slug}.json", asdict(card))
    workspace.write_text(
        f"logs/concept_enrichment/{card.slug}.txt",
        f"tavily_research={research_path}\nwhy_source={why_source}\nwhy_model={agent_model}\n",
    )
    if progress:
        progress("saving", "Saving the enriched concept card.")
    return card


def enrich_concept_card(
    parsed: ParsedPaper,
    workspace: Workspace,
    term: str,
    progress: Callable[[str, str], None] | None = None,
) -> ConceptCard:
    with workflow_trace(
        "concept-enrichment",
        input_data={"term": term, "paper": parsed.metadata.title_guess},
        session_id=paper_session_id(workspace),
        tags=["concept", "paper-reader", "tavily"],
        metadata={"termSlug": slugify(term)},
    ) as trace:
        card = _enrich_concept_card_impl(parsed, workspace, term, progress)
        succeeded = card.status == "enriched"
        trace.update(
            output=asdict(card),
            metadata={
                "status": card.status,
                "webSourceCount": len(card.web_sources),
                "generalExplanationModel": card.general_explanation_model,
                "whyItMattersModel": card.why_it_matters_model,
            },
            level="DEFAULT" if succeeded else "WARNING",
        )
        trace.score(
            "concept_enrichment_complete",
            1.0 if succeeded else 0.0,
            data_type="BOOLEAN",
        )
        trace.score("concept_web_sources", float(len(card.web_sources)))
        trace.score(
            "concept_paper_agent_succeeded",
            1.0 if card.why_it_matters_source == "paper_agent" else 0.0,
            data_type="BOOLEAN",
        )
        return card

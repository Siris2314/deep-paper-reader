from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import tool

from paper_agent.claim_ledger import add_claim, save_claim_ledger, validate_claims_json
from paper_agent.config import DEFAULT_OLLAMA_MODEL, RunConfig
from paper_agent.debug import (
    write_agent_messages_markdown,
    write_debug_json,
    write_exception,
    write_run_config,
)
from paper_agent.harness import (
    HarnessLimitExceeded,
    bounded_env,
    create_harness_agent,
    harness_workflow,
)
from paper_agent.observability import invoke_observed, paper_session_id, workflow_trace
from paper_agent.parser import ParsedPaper, parsed_artifact_manifest
from paper_agent.report_evaluator import (
    build_minimal_claim_ledger,
    build_repair_prompt,
    deterministic_fallback_report,
    evaluate_report_text,
    save_report_evaluation,
)
from paper_agent.schemas import ClaimRecord
from paper_agent.skills import (
    create_deep_agent_with_optional_skills,
    format_skill_manifest_for_prompt,
    skill_manifest,
)
from paper_agent.tavily_research import build_tavily_tools, run_tavily_prefetch
from paper_agent.term_memory import ensure_term_memory_files
from paper_agent.workspace import Workspace

SUPERVISOR_PROMPT = """
You are a lean Deep Research Paper Agent supervisor.

Main rule: do not hardcode every workflow into the main context. Use skills when the task needs a consistent multi-step workflow, portable domain expertise, lower token usage, or deterministic helper code. The main prompt only routes work, enforces paper-grounding, and manages workspace files.

Core rules:
- Treat the uploaded paper as the primary source of truth.
- Do not summarize the whole extracted text blob blindly.
- Prefer retrieved sections, parsed equations, figures, and tables over full-paper context.
- Separate paper evidence, web context, and inference.
- Do not invent claims, benchmarks, citations, URLs, equations, code, or implementation details.
- Use workspace files as external memory; do not keep large artifacts in conversation.
- Load detailed skill behavior only when the user/report needs it.
- Save intermediate files in scratch/ and final outputs in final/.

Skill-trigger examples:
- math/equation/derivation/notation question -> math-walkthrough skill.
- implementation/code/pseudocode/system question -> implementation-reconstructor skill.
- concept popup/background term -> concept-card skill.
- web/related work/prerequisite lookup -> tavily-background skill.
- final answer/report quality check -> paper-evaluator skill.

Required report outputs:
- scratch/summary_agent.md
- scratch/method_agent.md
- scratch/math_agent.md
- scratch/experiments_agent.md
- scratch/critic_agent.md
- final/final_study_guide.md
- final/claim_ledger.json

Final report sections:
TLDR; problem/motivation; main contributions; method; math walkthrough; figures/tables; experiments/results; related work if available; limitations; prerequisites; test questions; claim-checking notes.
""".strip()

SUMMARY_AGENT = {
    "name": "summary-agent",
    "description": "Creates TLDR, main points, and section-by-section notes using only paper artifacts.",
    "system_prompt": """
You summarize research papers. Use only the paper content provided by the supervisor or workspace tools.

Return:
1. One-sentence TLDR
2. Main problem
3. Main idea
4. 5-8 main contributions or claims
5. Section-by-section summary
6. Why the paper matters
7. What a reader should not overclaim

Do not use web context unless the supervisor explicitly provides it. Do not invent unsupported claims.
""".strip(),
}

METHOD_AGENT = {
    "name": "method-agent",
    "description": "Explains the proposed model, architecture, system pipeline, and algorithmic steps.",
    "system_prompt": """
You explain the method of a research paper.

Return:
1. Inputs and outputs
2. Algorithm/pipeline steps
3. Key design choices
4. Pseudocode if possible
5. How the method differs from prior/common methods
6. Failure modes and assumptions
7. Implementation details that are missing from the paper

Be concrete. If the paper does not provide details, mark them as missing.
""".strip(),
}

MATH_AGENT = {
    "name": "math-agent",
    "description": "Explains notation, equations, derivations, tensor shapes, and mathematical intuition.",
    "system_prompt": """
You explain the math in research papers.

Return equation cards. For each important equation include:
- equation id/page if available
- raw equation
- purpose
- variables and shapes when inferable
- plain-English meaning
- derivation or missing derivation note
- minimal toy example when useful

Also include:
1. Notation table
2. Core equations
3. Tensor/matrix shape interpretation
4. Assumptions
5. What the paper leaves unexplained

Be strict. Do not pretend an equation proves more than it does.
""".strip(),
}

EXPERIMENTS_AGENT = {
    "name": "experiments-agent",
    "description": "Analyzes datasets, baselines, metrics, tables, ablations, and diagnostic evidence.",
    "system_prompt": """
You analyze the experiments section of a research paper.

Return:
1. Datasets
2. Baselines
3. Metrics
4. Main results
5. Tables and figures worth citing
6. Ablations/diagnostics
7. Whether evidence supports the paper's claims
8. Weaknesses in evaluation

If experiments are absent or weak, say so directly. Do not round or alter table values unless you say you are approximating.
""".strip(),
}

CRITIC_AGENT = {
    "name": "critic-agent",
    "description": "Checks the final guide for unsupported, exaggerated, or mixed-source claims.",
    "system_prompt": """
You are a strict grounding and claim-checking agent.

Check generated claims against supplied paper artifacts and any supplied external context.

For important claims, label:
- supported
- partially_supported
- unsupported
- needs_external_verification

Flag:
- invented SOTA claims
- vague benchmark claims
- overgeneralized results
- missing limitations
- claims that come from web context rather than the paper

Return a concise critique plus a corrected claim ledger in JSON if possible.
""".strip(),
}

EVALUATOR_AGENT = {
    "name": "evaluator-agent",
    "description": "Evaluates whether the generated study guide is actually about the paper and flags drift into appendices or generic summaries.",
    "system_prompt": """
You are a final-report evaluator for a paper-analysis system.

Check whether the study guide is actually about the paper's title, abstract, method, experiments, figures, and tables.

Flag these problems hard:
1. The report summarizes appendix/system prompt text instead of the paper's main contribution.
2. The report starts with generic wording like 'The provided text appears...'.
3. The report ignores the title and abstract.
4. The report ignores central figures/tables/results.
5. The report has an empty or unsupported claim ledger.

Return:
- pass/fail
- concise reasons
- specific repairs required
- corrected outline

Be strict. A fluent but wrong report fails.
""".strip(),
}


RELATED_WORK_AGENT = {
    "name": "related-work-agent",
    "description": "Uses Tavily web tools to find official pages, code, related papers, and background context.",
    "system_prompt": """
You are a related-work and external-context research agent.

Use web tools only for external context. Prefer official sources:
- arXiv
- OpenReview
- ACL Anthology
- NeurIPS / ICML / ICLR / CVPR / ICCV / ECCV pages
- Hugging Face model/dataset pages
- GitHub repositories
- author/project pages

Return:
1. Official paper/page links found
2. Code/project links found
3. Related papers or methods
4. Background concepts useful for understanding the paper
5. What each source adds
6. Conflicts or uncertainty

Do not let web context override the paper unless paper metadata is clearly outdated.
""".strip(),
}


def build_chat_model(config: RunConfig):
    provider = config.model_provider.lower().strip()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama support requires langchain-ollama. Run: pip install langchain-ollama"
            ) from exc

        model_name = config.model
        if model_name.startswith("openai:"):
            model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=0.1,
            num_predict=bounded_env("PAPER_HARNESS_MAX_OUTPUT_TOKENS", 1800, 128, 8192),
            keep_alive=os.getenv("PAPER_READER_AGENT_KEEP_ALIVE", "15m"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            client_kwargs={
                "timeout": max(10.0, min(float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120")), 600.0))
            },
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires langchain-openai. Run: pip install langchain-openai"
            ) from exc
        model_name = config.model.replace("openai:", "")
        return ChatOpenAI(model=model_name, temperature=0.1)

    raise ValueError(f"Unsupported model provider: {config.model_provider}")


def build_direct_chat_model(config: RunConfig):
    """Return a chat model object that supports .invoke for post-hoc repair.

    Deep Agents accepts provider strings for some backends, but the evaluator repair loop
    needs a direct LangChain chat object.
    """
    provider = config.model_provider.lower().strip()
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama support requires langchain-ollama. Run: pip install langchain-ollama"
            ) from exc
        model_name = config.model
        if model_name.startswith("openai:"):
            model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=0.0,
            num_predict=bounded_env("PAPER_HARNESS_MAX_OUTPUT_TOKENS", 1800, 128, 8192),
            keep_alive=os.getenv("PAPER_READER_AGENT_KEEP_ALIVE", "15m"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            client_kwargs={
                "timeout": max(10.0, min(float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120")), 600.0))
            },
        )
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires langchain-openai. Run: pip install langchain-openai"
            ) from exc
        model_name = config.model.replace("openai:", "")
        return ChatOpenAI(model=model_name, temperature=0.0)
    raise ValueError(f"Unsupported model provider: {config.model_provider}")


def invoke_text_model(
    model: Any,
    prompt: str,
    *,
    name: str = "direct-model-generation",
    model_name: str | None = None,
) -> str:
    response = invoke_observed(model, prompt, name=name, model=model_name)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    if isinstance(response, dict):
        return str(response.get("content", response))
    return str(response)


def _retrieval_slice(text: str, max_chars: int, offset: int) -> str:
    offset = max(0, offset)
    size = max(1, min(max_chars, 6000))
    end = min(len(text), offset + size)
    result = text[offset:end]
    if end < len(text):
        result += f"\n[More available: offset={end}, total_chars={len(text)}]"
    return result


def _artifact_page(cards: list, offset: int, limit: int) -> str:
    offset = max(0, offset)
    end = min(len(cards), offset + max(1, min(limit, 12)))
    return json.dumps(
        {
            "cards": [card.model_dump() for card in cards[offset:end]],
            "next_offset": end if end < len(cards) else None,
            "total": len(cards),
        },
        ensure_ascii=False,
    )


def build_workspace_tools(workspace: Workspace, parsed: ParsedPaper):
    paper_text = parsed.full_text
    sections = parsed.sections

    @tool
    def save_workspace_file(rel_path: str, content: str) -> str:
        """Save text into the paper_report workspace using a relative path."""
        try:
            path = workspace.write_text(rel_path, content)
            return f"Saved {path}"
        except Exception as exc:
            return f"ERROR saving file: {exc}"

    @tool
    def read_workspace_file(rel_path: str, max_chars: int = 6000, offset: int = 0) -> str:
        """Read a bounded character slice of a workspace file; use offset for the next slice."""
        try:
            return _retrieval_slice(workspace.read_text(rel_path), max_chars, offset)
        except Exception as exc:
            return f"ERROR reading file: {exc}"

    @tool
    def list_workspace_files() -> str:
        """List files currently saved in the workspace."""
        return "\n".join(workspace.list_files()) or "No files found."

    @tool
    def grep_workspace(pattern: str, max_hits: int = 20) -> str:
        """Search workspace text files for a regex pattern."""
        try:
            rx = re.compile(pattern, re.I)
        except re.error as exc:
            return f"Invalid regex: {exc}"
        hits: list[str] = []
        for rel in workspace.list_files():
            if not rel.endswith((".md", ".txt", ".json")):
                continue
            text = workspace.read_text(rel, max_chars=200_000)
            for line_no, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{rel}:{line_no}: {line[:400]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits)
        return "No matches."

    @tool
    def search_paper_text(query: str, max_hits: int = 8) -> str:
        """Search the extracted paper text for direct matches."""
        query_lower = query.lower().strip()
        if not query_lower:
            return "Empty query."
        chunks = paper_text.split("\n\n")
        hits: list[str] = []
        for chunk in chunks:
            if query_lower in chunk.lower():
                hits.append(chunk.strip()[:2000])
            if len(hits) >= max_hits:
                break
        return "\n\n--- HIT ---\n\n".join(hits) if hits else "No direct text matches found."

    @tool
    def get_section(section_name: str, max_chars: int = 6000, offset: int = 0) -> str:
        """Read a section slice by name; use the returned offset to continue long sections."""
        key = section_name.lower().strip().replace(" ", "_").replace("/", "_")
        if key in sections:
            text = sections[key]
            return _retrieval_slice(text, max_chars, offset)
        available = ", ".join(sections.keys())
        return f"Section not found. Available sections: {available}"

    @tool
    def get_equation_cards(offset: int = 0, limit: int = 8) -> str:
        """Return a bounded page of equation cards as JSON, with the next card offset."""
        return _artifact_page(parsed.equation_cards, offset, limit)

    @tool
    def get_figure_cards(offset: int = 0, limit: int = 8) -> str:
        """Return a bounded page of figure captions as JSON, with the next card offset."""
        return _artifact_page(parsed.figure_cards, offset, limit)

    @tool
    def get_table_cards(offset: int = 0, limit: int = 8) -> str:
        """Return a bounded page of table cards as JSON, with the next card offset."""
        return _artifact_page(parsed.table_cards, offset, limit)

    @tool
    def add_claim_to_ledger(
        claim: str,
        source_type: str,
        support: str,
        evidence: str,
        confidence: str = "medium",
    ) -> str:
        """Add one atomic claim to final/claim_ledger.json."""
        try:
            record = ClaimRecord(
                claim=claim,
                source_type=source_type,  # type: ignore[arg-type]
                support=support,  # type: ignore[arg-type]
                evidence=evidence,
                confidence=confidence,  # type: ignore[arg-type]
            )
            path = add_claim(workspace, record)
            return f"Added claim to {path}"
        except Exception as exc:
            return f"ERROR adding claim: {exc}"

    @tool
    def save_claim_ledger_json(raw_json: str) -> str:
        """Validate and save the complete final claim ledger JSON list."""
        ok, msg = validate_claims_json(raw_json)
        if not ok:
            return f"INVALID CLAIM LEDGER: {msg}"
        records = [ClaimRecord.model_validate(item) for item in json.loads(raw_json)]
        path = save_claim_ledger(workspace, records)
        return f"Saved validated claim ledger to {path}"

    return [
        save_workspace_file,
        read_workspace_file,
        list_workspace_files,
        grep_workspace,
        search_paper_text,
        get_section,
        get_equation_cards,
        get_figure_cards,
        get_table_cards,
        add_claim_to_ledger,
        save_claim_ledger_json,
    ]


def build_agent(config: RunConfig, parsed: ParsedPaper, workspace: Workspace):
    local_tools = build_workspace_tools(workspace, parsed)
    tavily_tools = build_tavily_tools(config.research_enabled)

    subagents = [
        SUMMARY_AGENT,
        METHOD_AGENT,
        MATH_AGENT,
        EXPERIMENTS_AGENT,
        CRITIC_AGENT,
        EVALUATOR_AGENT,
    ]
    if tavily_tools:
        related = dict(RELATED_WORK_AGENT)
        related["tools"] = tavily_tools
        subagents.append(related)

    chat_model = build_chat_model(config)
    _, learned_terms_memory = ensure_term_memory_files()
    project_memory = Path(__file__).resolve().parents[2] / "AGENTS.md"

    return create_harness_agent(
        lambda **kwargs: create_deep_agent_with_optional_skills(create_deep_agent, **kwargs),
        model=chat_model,
        tools=[*local_tools, *tavily_tools],
        system_prompt=SUPERVISOR_PROMPT,
        subagents=subagents,
        memory=[str(project_memory), str(learned_terms_memory)],
    )


def _lean_paper_bundle(parsed: ParsedPaper, max_chars: int = 12_000) -> str:
    section_names = ", ".join(parsed.sections) or "none detected"
    abstract = parsed.sections.get("abstract", "").strip()
    if not abstract:
        abstract = parsed.page_text.get(1, "").strip()
    equation_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.raw[:220]}" for card in parsed.equation_cards[:8]
    )
    figure_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.caption[:220]}" for card in parsed.figure_cards[:6]
    )
    table_preview = "\n".join(
        f"- {card.id} page {card.page}: {card.caption[:220]}" for card in parsed.table_cards[:6]
    )
    bundle = f"""
Title: {parsed.metadata.title_guess or "Untitled paper"}
Pages: {parsed.metadata.page_count}
Detected sections: {section_names}

Abstract or first-page overview:
{abstract[:4500] or "Unavailable."}

Equation index:
{equation_preview or "No equation candidates detected."}

Figure index:
{figure_preview or "No figure captions detected."}

Table index:
{table_preview or "No table captions detected."}

This is an index, not the paper body. Retrieve relevant sections and artifacts with workspace tools
before making claims; do not infer omitted method or result details from this bundle.
""".strip()
    return bundle[:max_chars]


def build_user_prompt(config: RunConfig, parsed: ParsedPaper, workspace: Workspace) -> str:
    tavily_status = "disabled"
    tavily_status_path = workspace.path("web/tavily_status.json")
    if tavily_status_path.exists():
        try:
            tavily_status_payload = json.loads(tavily_status_path.read_text(encoding="utf-8"))
            tavily_status = f"{tavily_status_payload.get('reason')} ({tavily_status_payload.get('result_count', 0)} sources)"
        except Exception:
            tavily_status = "status file unreadable"
    elif build_tavily_tools(config.research_enabled):
        tavily_status = "enabled as agent tools only"
    if config.research_enabled and not os.getenv("TAVILY_API_KEY"):
        tavily_status = "requested, but disabled because TAVILY_API_KEY is missing"

    artifact_manifest = json.dumps(parsed_artifact_manifest(parsed), indent=2, ensure_ascii=False)
    paper_bundle = _lean_paper_bundle(parsed)

    strict_note = (
        "Use final/claim_ledger.json schema strictly."
        if config.strict_json
        else "JSON strictness relaxed."
    )

    skills = format_skill_manifest_for_prompt()

    return f"""
Analyze this research paper and create a full study guide.

Run settings:
- mode: {config.mode}
- model provider: {config.model_provider}
- model: {config.model}
- Tavily/web research: {tavily_status}
- workspace root: {workspace.root}
- strict JSON: {config.strict_json}

Workspace artifact manifest:
{artifact_manifest}

Available skills (full instructions load only when needed):
{skills}

Required behavior:
- Use parsed files when possible instead of keeping everything in context.
- If web/tavily_sources.json exists, read it and use it for the external-context section.
- Use get_figure_cards and get_table_cards; the final report must discuss important figures/tables.
- Use get_equation_cards; the math section must be equation-card based.
- Save final report to final/final_study_guide.md.
- Save claim ledger to final/claim_ledger.json using save_claim_ledger_json.
- {strict_note}
- If Tavily is disabled, leave external context blank rather than hallucinating it.

Initial paper bundle:
{paper_bundle}
""".strip()


def extract_final_message(result: Any) -> str:
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last = result["messages"][-1]
        content = getattr(last, "content", None)
        if content is not None:
            return str(content)
        if isinstance(last, dict):
            return str(last.get("content", last))
    return str(result)


def _run_analysis_impl(config: RunConfig, parsed: ParsedPaper) -> str:
    workspace = Workspace(config.output_dir)
    write_run_config(workspace, config)
    workspace.write_json("logs/skill_manifest.json", skill_manifest())

    if config.no_agent:
        msg = (
            "Parse-only run completed. Parsed artifacts were written to the workspace. "
            "Run without --no-agent to generate the study guide."
        )
        workspace.write_text("final/final_study_guide.md", msg)
        return msg

    if config.research_enabled:
        prefetch = run_tavily_prefetch(
            parsed=parsed,
            workspace=workspace,
            enabled=True,
            max_results=config.tavily_max_results,
        )
        workspace.write_text(
            "logs/tavily_prefetch.txt",
            f"enabled={prefetch.enabled} attempted={prefetch.attempted} success={prefetch.success} "
            f"sources={prefetch.result_count} reason={prefetch.reason}\n"
            f"queries={prefetch.queries}\n",
        )
    else:
        run_tavily_prefetch(parsed=parsed, workspace=workspace, enabled=False)

    agent = build_agent(config, parsed, workspace)
    prompt = build_user_prompt(config, parsed, workspace)
    workspace.write_text("logs/user_prompt.md", prompt)

    try:
        result = invoke_observed(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            name="full-paper-agent",
            model=f"{config.model_provider}:{config.model}",
            metadata={"mode": config.mode},
        )
    except Exception as exc:
        write_exception(workspace, exc)
        raise

    write_debug_json(workspace, "logs/agent_result.json", result)
    write_agent_messages_markdown(workspace, result)
    final = extract_final_message(result)
    workspace.write_text("logs/final_message.txt", final)

    final_path = workspace.path("final/final_study_guide.md")
    if not final_path.exists() or not final_path.read_text(encoding="utf-8").strip():
        workspace.write_text("final/final_study_guide.md", final)

    # Backward-compatible flat report path.
    flat_final = workspace.path("final_study_guide.md")
    if not flat_final.exists():
        flat_final.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")

    claim_path = workspace.path("final/claim_ledger.json")
    if not claim_path.exists():
        # Empty but valid ledger. The critic is expected to fill this; this prevents missing-file failure.
        save_claim_ledger(workspace, [])

    if config.run_evaluator:
        final_text = final_path.read_text(encoding="utf-8", errors="replace")
        claim_text = (
            claim_path.read_text(encoding="utf-8", errors="replace")
            if claim_path.exists()
            else "[]"
        )
        evaluation = evaluate_report_text(parsed, final_text, claim_text)
        save_report_evaluation(workspace, evaluation)

        if not evaluation.passed and config.auto_repair_attempts > 0:
            workspace.write_text(
                "logs/evaluator_status.txt", "initial evaluation failed; attempting repair\n"
            )
            repair_prompt = build_repair_prompt(parsed, final_text, evaluation, workspace)
            workspace.write_text("scratch/evaluator_repair_prompt.md", repair_prompt)
            try:
                repair_model = build_direct_chat_model(config)
                repaired = invoke_text_model(
                    repair_model,
                    repair_prompt,
                    name="report-repair-generation",
                    model_name=f"{config.model_provider}:{config.model}",
                ).strip()
                workspace.write_text("logs/evaluator_repair_raw.txt", repaired)
                workspace.write_text("final/final_study_guide.repaired.md", repaired + "\n")
                repaired_eval = evaluate_report_text(parsed, repaired, claim_text)
                save_report_evaluation(
                    workspace, repaired_eval, name="final/report_evaluation_repaired.json"
                )
                if repaired_eval.score >= evaluation.score and not any(
                    i.severity == "fatal" for i in repaired_eval.issues
                ):
                    workspace.write_text("final/final_study_guide.md", repaired + "\n")
                    evaluation = repaired_eval
                    save_report_evaluation(workspace, evaluation)
                    workspace.write_text("logs/evaluator_status.txt", "repair accepted\n")
                else:
                    workspace.write_text(
                        "logs/evaluator_status.txt",
                        "repair rejected; generated deterministic fallback because report still failed fatal checks\n",
                    )
                    fallback = deterministic_fallback_report(parsed, evaluation)
                    workspace.write_text("final/final_study_guide.fallback.md", fallback)
                    workspace.write_text("final/final_study_guide.md", fallback)
                    workspace.write_json(
                        "final/claim_ledger.json", build_minimal_claim_ledger(parsed)
                    )
                    final_text = fallback
                    claim_text = json.dumps(build_minimal_claim_ledger(parsed), indent=2)
                    fallback_eval = evaluate_report_text(parsed, final_text, claim_text)
                    save_report_evaluation(workspace, fallback_eval)
            except HarnessLimitExceeded:
                raise
            except Exception as exc:
                write_exception(workspace, exc)
                fallback = deterministic_fallback_report(parsed, evaluation)
                workspace.write_text("final/final_study_guide.fallback.md", fallback)
                workspace.write_text("final/final_study_guide.md", fallback)
                workspace.write_json("final/claim_ledger.json", build_minimal_claim_ledger(parsed))
                workspace.write_text(
                    "logs/evaluator_status.txt", f"repair exception; fallback generated: {exc}\n"
                )
        else:
            workspace.write_text(
                "logs/evaluator_status.txt", f"evaluation {evaluation.status}; no repair needed\n"
            )

    # Keep backward-compatible flat report synchronized after evaluator/repair/fallback.
    workspace.path("final_study_guide.md").write_text(
        final_path.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )
    return final_path.read_text(encoding="utf-8")


@harness_workflow("report")
def run_analysis(config: RunConfig, parsed: ParsedPaper) -> str:
    workspace = Workspace(config.output_dir)
    with workflow_trace(
        "full-paper-analysis",
        input_data={
            "paper": parsed.metadata.title_guess,
            "mode": config.mode,
            "provider": config.model_provider,
            "model": config.model,
        },
        session_id=paper_session_id(workspace),
        tags=["report", "paper-reader", config.mode],
        metadata={
            "mode": config.mode,
            "provider": config.model_provider,
            "model": config.model,
            "evaluatorEnabled": config.run_evaluator,
        },
    ) as trace:
        final = _run_analysis_impl(config, parsed)
        evaluation_path = workspace.path("final/report_evaluation.json")
        evaluation: dict[str, Any] = {}
        if evaluation_path.exists():
            try:
                loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
                evaluation = loaded if isinstance(loaded, dict) else {}
            except Exception:
                evaluation = {}
        trace.update(
            output={"report": final},
            metadata={
                "reportCharacters": len(final),
                "evaluationStatus": evaluation.get("status", "unavailable"),
            },
            level="DEFAULT" if evaluation.get("passed", True) else "WARNING",
        )
        if evaluation:
            trace.score(
                "report_passed",
                1.0 if evaluation.get("passed") else 0.0,
                data_type="BOOLEAN",
            )
            trace.score("report_score", float(evaluation.get("score") or 0))
        return final

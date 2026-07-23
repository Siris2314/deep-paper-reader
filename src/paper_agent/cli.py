from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from paper_agent.config import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TAVILY_MAX_RESULTS,
    RunConfig,
)
from paper_agent.parser import parse_paper, parsed_artifact_manifest
from paper_agent.tavily_research import run_tavily_prefetch
from paper_agent.workspace import Workspace

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-agent",
        description="Parse, inspect, and analyze research papers with skills and grounded agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a local research paper PDF.")
    analyze.add_argument("--pdf", required=True, help="Path to the local PDF file.")
    analyze.add_argument(
        "--mode",
        choices=["paper-only", "research"],
        default=os.getenv("RUN_MODE", "paper-only"),
        help="paper-only uses only the PDF; research also uses Tavily if configured.",
    )
    analyze.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=os.getenv("MODEL_PROVIDER", "ollama"),
        help="Model backend. Use ollama for local models.",
    )
    analyze.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", DEFAULT_OLLAMA_MODEL),
        help=(
            "Model name. For OpenAI use provider:model, e.g. openai:gpt-5.5. "
            f"For Ollama use local model name, e.g. {DEFAULT_OLLAMA_MODEL}."
        ),
    )
    analyze.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        help=f"Ollama server URL. Default: {DEFAULT_OLLAMA_BASE_URL}",
    )
    analyze.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    analyze.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("MAX_CHARS", str(DEFAULT_MAX_CHARS))),
        help="Maximum extracted paper characters passed in the initial prompt.",
    )
    analyze.add_argument(
        "--use-tavily",
        action="store_true",
        help="Force Tavily tools on if TAVILY_API_KEY is set. Equivalent to research web tools.",
    )
    analyze.add_argument(
        "--tavily-max-results",
        type=int,
        default=int(os.getenv("TAVILY_MAX_RESULTS", str(DEFAULT_TAVILY_MAX_RESULTS))),
        help="Max Tavily results per query for deterministic research prefetch.",
    )
    analyze.add_argument(
        "--no-agent",
        action="store_true",
        help="Parse the PDF and write artifacts without invoking the LLM agent.",
    )
    analyze.add_argument(
        "--no-strict-json",
        action="store_true",
        help="Relax final claim ledger JSON validation instructions.",
    )
    analyze.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear the output workspace before this run. By default each run clears generated artifacts to avoid stale reports.",
    )
    analyze.add_argument(
        "--no-evaluator",
        action="store_true",
        help="Disable the post-run report evaluator and auto-repair loop.",
    )
    analyze.add_argument(
        "--auto-repair-attempts",
        type=int,
        default=int(os.getenv("AUTO_REPAIR_ATTEMPTS", "1")),
        help="Number of evaluator repair attempts after a failed report-quality check. Default: 1.",
    )

    inspect = subparsers.add_parser("inspect", help="Inspect an existing paper_report workspace.")
    inspect.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Workspace directory.")
    inspect.add_argument("--file", default=None, help="Optional workspace file to print.")
    inspect.add_argument("--max-chars", type=int, default=8000)

    concepts = subparsers.add_parser(
        "concepts", help="Extract or inspect concept cards for a parsed workspace."
    )
    concepts.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Workspace directory.")
    concepts.add_argument("--term", default=None, help="Create/show a concept card for this term.")

    chat = subparsers.add_parser("chat", help="Ask one question against a parsed paper workspace.")
    chat.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Workspace directory.")
    chat.add_argument("--question", required=True, help="Question to ask about the parsed paper.")
    chat.add_argument(
        "--provider", choices=["openai", "ollama"], default=os.getenv("MODEL_PROVIDER", "ollama")
    )
    chat.add_argument("--model", default=os.getenv("MODEL_NAME", DEFAULT_OLLAMA_MODEL))
    chat.add_argument(
        "--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )

    hardware = subparsers.add_parser(
        "hardware", help="Detect hardware and select a safe local inference profile."
    )
    hardware.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    )
    hardware.add_argument(
        "--profile",
        choices=["auto", "cpu", "low-vram", "balanced", "high-vram"],
        default="auto",
    )
    hardware.add_argument(
        "--apply", action="store_true", help="Update managed model settings in .env."
    )
    hardware.add_argument(
        "--apply-server",
        action="store_true",
        help="Apply the selected Ollama server profile on Windows.",
    )
    hardware.add_argument("--json", action="store_true", help="Print the hardware report as JSON.")

    return parser


def analyze_command(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if args.provider == "ollama" and args.model.startswith("openai:"):
        args.model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if args.provider == "openai" and not args.model.startswith("openai:"):
        args.model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    config = RunConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        mode=args.mode,
        model_provider=args.provider,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        max_chars=args.max_chars,
        use_tavily=bool(args.use_tavily or args.mode == "research"),
        strict_json=not args.no_strict_json,
        no_agent=args.no_agent,
        tavily_max_results=args.tavily_max_results,
        clean_output=not args.no_clean,
        run_evaluator=not args.no_evaluator,
        auto_repair_attempts=args.auto_repair_attempts,
    )
    config.validate()

    workspace = Workspace(config.output_dir)
    if config.clean_output:
        console.print(Panel.fit(f"Clearing workspace: {config.output_dir}", border_style="yellow"))
        workspace.reset()

    console.print(Panel.fit("Parsing PDF", border_style="cyan"))
    parsed = parse_paper(config.pdf_path, config.output_dir)
    try:
        from paper_agent.concepts import save_concept_index

        save_concept_index(parsed, workspace)
    except Exception:
        pass
    console.print(json.dumps(parsed_artifact_manifest(parsed), indent=2, ensure_ascii=False))

    if config.no_agent:
        if config.research_enabled:
            workspace = Workspace(config.output_dir)
            prefetch = run_tavily_prefetch(
                parsed=parsed,
                workspace=workspace,
                enabled=True,
                max_results=config.tavily_max_results,
            )
            console.print(
                Panel.fit(
                    f"Tavily prefetch: attempted={prefetch.attempted}, success={prefetch.success}, "
                    f"sources={prefetch.result_count}\nReason: {prefetch.reason}",
                    border_style="yellow" if not prefetch.success else "green",
                )
            )
        workspace = Workspace(config.output_dir)
        workspace.write_text(
            "final/final_study_guide.md",
            (
                f"Parse-only run completed for `{config.pdf_path}`. "
                "No model summary was generated in this run. Use `Run full agent` "
                "or remove `--no-agent` to create the study guide.\n"
            ),
        )
        workspace.write_text("final/claim_ledger.json", "[]\n")
        console.print(Panel.fit(f"Parse-only complete: {config.output_dir}", border_style="green"))
        return 0

    console.print(
        Panel.fit(
            f"Running Deep Agent via {config.model_provider}: {config.model}",
            border_style="cyan",
        )
    )
    from paper_agent.agents import run_analysis

    final = run_analysis(config, parsed)

    final_path = config.output_dir / "final" / "final_study_guide.md"
    claim_path = config.output_dir / "final" / "claim_ledger.json"
    console.print(
        Panel.fit(
            f"Done\n\nFinal report: {final_path}\nClaim ledger: {claim_path}", border_style="green"
        )
    )
    console.print(final[:4000])
    if len(final) > 4000:
        console.print("\n[dim]Output truncated in terminal. Open final/final_study_guide.md.[/dim]")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    workspace = Workspace(args.output)
    if args.file:
        console.print(workspace.read_text(args.file, max_chars=args.max_chars))
        return 0
    console.print("\n".join(workspace.list_files()) or "No files found.")
    return 0


def concepts_command(args: argparse.Namespace) -> int:
    from paper_agent.concepts import (
        save_concept_index,
        get_or_create_concept_card,
        concept_card_to_markdown,
    )
    from paper_agent.parser import load_parsed_paper_from_workspace

    workspace = Workspace(args.output)
    parsed = load_parsed_paper_from_workspace(workspace)
    concepts = save_concept_index(parsed, workspace)
    if args.term:
        card = get_or_create_concept_card(parsed, workspace, args.term)
        md = concept_card_to_markdown(card)
        workspace.write_text(f"concepts/cards/{card.slug}.md", md)
        console.print(md)
    else:
        console.print("\n".join(concepts))
    return 0


def chat_command(args: argparse.Namespace) -> int:
    from paper_agent.chat_agent import chat_once

    output_dir = Path(args.output).expanduser().resolve()
    # source_pdf is only used for config bookkeeping in chat mode; parsed artifacts are loaded from workspace.
    metadata_path = output_dir / "parsed" / "metadata.json"
    pdf_path = Path("paper.pdf")
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            pdf_path = Path(payload.get("source_pdf") or "paper.pdf")
        except Exception:
            pass
    config = RunConfig(
        pdf_path=pdf_path,
        output_dir=output_dir,
        mode="paper-only",
        model_provider=args.provider,
        model=args.model,
        ollama_base_url=args.ollama_base_url,
        clean_output=False,
    )
    answer = chat_once(config, args.question)
    console.print(answer)
    return 0


def hardware_command(args: argparse.Namespace) -> int:
    from paper_agent.hardware_setup import (
        apply_project_profile,
        apply_windows_server_profile,
        detect_hardware,
        hardware_report,
        print_report,
        select_profile,
    )

    hardware = detect_hardware(args.base_url)
    profile = select_profile(hardware, args.profile)
    report = hardware_report(hardware, profile)
    if args.json:
        console.print_json(data=report)
    else:
        print_report(report)
    if args.apply:
        path = apply_project_profile(hardware, profile)
        if not args.json:
            console.print(f"\nProject profile saved to {path}")
    if args.apply_server:
        apply_windows_server_profile(profile)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return analyze_command(args)
    if args.command == "inspect":
        return inspect_command(args)
    if args.command == "concepts":
        return concepts_command(args)
    if args.command == "chat":
        return chat_command(args)
    if args.command == "hardware":
        return hardware_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

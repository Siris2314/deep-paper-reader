from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from paper_agent.agents import build_chat_model
from paper_agent.config import DEFAULT_OLLAMA_BASE_URL, RunConfig
from paper_agent.context_diagnostics import record_context_usage, response_schema_text
from paper_agent.harness import HarnessLimitExceeded, create_harness_agent, harness_workflow
from paper_agent.judge_memory import retrieve_math_judge_memory
from paper_agent.observability import invoke_observed, observed_stage
from paper_agent.workspace import Workspace


@dataclass
class JudgeDimension:
    score: float
    justification: str


@dataclass
class MathJudgeResult:
    verdict: str
    overall_score: float
    dimensions: dict[str, dict[str, object]]
    summary: str
    issues: list[str]
    deterministic_checks: dict[str, object]
    judge_model: str
    evaluated_at: str


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class MathJudgePayload(BaseModel):
    correctness: JudgeScore
    paper_grounding: JudgeScore
    symbol_coverage: JudgeScore
    latex_fidelity: JudgeScore
    usefulness: JudgeScore
    summary: str
    issues: list[str] = Field(default_factory=list)


_INLINE_MATH_FIELDS = (
    "plain_english",
    "intuition",
    "dimensional_analysis",
    "implementation_view",
    "context_fit",
)
_SUPPORTED_INLINE_COMMANDS = {
    "alpha",
    "approx",
    "argmax",
    "argmin",
    "bar",
    "beta",
    "bigcup",
    "cdot",
    "cup",
    "dots",
    "ell",
    "exp",
    "exists",
    "frac",
    "gamma",
    "geq",
    "hat",
    "in",
    "infty",
    "ldots",
    "leq",
    "left",
    "limits",
    "log",
    "lvert",
    "mathbb",
    "mathbf",
    "mathcal",
    "max",
    "mathrm",
    "middle",
    "mid",
    "min",
    "mu",
    "nabla",
    "ne",
    "neq",
    "notin",
    "nu",
    "omega",
    "operatorname",
    "overline",
    "partial",
    "pi",
    "phi",
    "prod",
    "propto",
    "psi",
    "quad",
    "qquad",
    "rho",
    "right",
    "rightarrow",
    "rvert",
    "sigma",
    "sim",
    "sqrt",
    "subset",
    "subseteq",
    "sum",
    "sup",
    "tau",
    "text",
    "textstyle",
    "theta",
    "tilde",
    "times",
    "to",
    "underbrace",
    "underline",
    "vert",
    "forall",
}


def _inline_math_issues(explanation: dict[str, Any]) -> list[str]:
    field_values = [(name, str(explanation.get(name) or "")) for name in _INLINE_MATH_FIELDS]
    for name in ("steps", "assumptions_or_missing_details"):
        items = explanation.get(name)
        if isinstance(items, list):
            field_values.extend((name, str(item or "")) for item in items)

    issues: list[str] = []
    for field_name, field_text in field_values:
        segments = [
            match.group(1) or match.group(2) or ""
            for match in re.finditer(r"\$([\s\S]*?)\$|\\\(([\s\S]*?)\\\)", field_text)
        ]
        for segment in segments:
            if re.search(r"[\x00-\x1f]", segment):
                issues.append("Inline math contains a decoded JSON control character.")
            if segment.count("{") != segment.count("}"):
                issues.append("Inline math has unbalanced braces.")
            if re.search(r"(?<![A-Za-z\\])(?:ext|imes|oldsymbol|rac)(?:\{|\b)", segment):
                issues.append(f"Inline math contains a damaged LaTeX command: {segment[:100]}")
            unsupported = sorted(
                {
                    command
                    for command in re.findall(r"\\([A-Za-z]+)", segment)
                    if command not in _SUPPORTED_INLINE_COMMANDS
                }
            )
            if unsupported:
                issues.append(
                    "Inline math uses unsupported commands: "
                    + ", ".join(f"\\{command}" for command in unsupported)
                )
        prose = re.sub(r"```[\s\S]*?```|`[^`\n]*`", " ", field_text)
        prose = re.sub(r"\$[\s\S]*?\$|\\\([\s\S]*?\\\)", " ", prose)
        bare_commands = sorted(
            {
                command
                for command in re.findall(r"\\([A-Za-z]+)", prose)
                if command in _SUPPORTED_INLINE_COMMANDS
            }
        )
        if bare_commands:
            issues.append(
                f"{field_name} contains LaTeX outside math delimiters: "
                + ", ".join(f"\\{command}" for command in bare_commands)
            )
    return list(dict.fromkeys(issues))[:12]


def _is_positive_number(value: object) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def deterministic_math_checks(explanation: dict[str, Any]) -> dict[str, object]:
    latex = str(explanation.get("display_latex") or "").strip()
    layout = (
        explanation.get("layout_evidence")
        if isinstance(explanation.get("layout_evidence"), dict)
        else {}
    )
    geometry_latex = str(layout.get("geometry_latex") or "").strip()
    crop_provenance = (
        layout.get("crop_provenance") if isinstance(layout.get("crop_provenance"), dict) else {}
    )
    symbols = explanation.get("symbols") if isinstance(explanation.get("symbols"), list) else []
    required_text = [
        "role",
        "plain_english",
        "intuition",
        "dimensional_analysis",
        "implementation_view",
        "context_fit",
    ]
    missing_fields = [
        name for name in required_text if not str(explanation.get(name) or "").strip()
    ]
    if not explanation.get("steps"):
        missing_fields.append("steps")
    if not explanation.get("paper_evidence"):
        missing_fields.append("paper_evidence")
    if not symbols:
        missing_fields.append("symbols")
    unsupported_sources = [
        str(item.get("symbol") or "")
        for item in symbols
        if isinstance(item, dict)
        and str(item.get("source") or "") not in {"paper", "inference", "unresolved"}
    ]
    dangerous_latex = bool(
        re.search(r"\\(?:href|url|includegraphics|input|write18)\b", latex, re.I)
    )
    inline_math_issues = _inline_math_issues(explanation)
    confidence_value = explanation.get("transcription_confidence")
    transcription_confidence = (
        float(confidence_value) if isinstance(confidence_value, (int, float)) else None
    )
    transcription_warnings = explanation.get("transcription_warnings")
    if not isinstance(transcription_warnings, list):
        transcription_warnings = []
    return {
        "required_fields_present": not missing_fields,
        "missing_fields": missing_fields,
        "latex_present": bool(latex),
        "latex_braces_balanced": latex.count("{") == latex.count("}"),
        "latex_commands_safe": not dangerous_latex,
        "geometry_latex_available": bool(geometry_latex),
        "geometry_latex_matches": bool(geometry_latex) and latex == geometry_latex,
        "vision_latex_used": explanation.get("latex_source") == "vision_math_agent",
        "vision_crop_grounded": (
            explanation.get("latex_source") != "vision_math_agent"
            or (
                bool(crop_provenance.get("sha256"))
                and _is_positive_number(crop_provenance.get("pixelWidth"))
                and _is_positive_number(crop_provenance.get("pixelHeight"))
            )
        ),
        "transcription_candidate_count": len(
            explanation.get("transcription_candidates")
            if isinstance(explanation.get("transcription_candidates"), dict)
            else {}
        ),
        "transcription_confidence": transcription_confidence,
        "transcription_confident": (
            transcription_confidence is None or transcription_confidence >= 0.6
        ),
        "transcription_warnings": [str(item) for item in transcription_warnings[:6]],
        "symbol_count": len(symbols),
        "symbol_meanings_complete": bool(symbols)
        and all(
            isinstance(item, dict)
            and bool(str(item.get("symbol") or "").strip())
            and bool(str(item.get("meaning") or "").strip())
            for item in symbols
        ),
        "unsupported_symbol_sources": unsupported_sources,
        "inline_math_valid": not inline_math_issues,
        "inline_math_issues": inline_math_issues,
    }


def _structured_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return MathJudgePayload.model_validate(result).model_dump()
    structured = result.get("structured_response") if isinstance(result, dict) else None
    if hasattr(structured, "model_dump"):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    messages = result.get("messages", []) if isinstance(result, dict) else []
    content = getattr(messages[-1], "content", "") if messages else ""
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    text = re.sub(r"<think>.*?</think>", "", str(content), flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge did not return structured output or JSON content.")
    return MathJudgePayload.model_validate(json.loads(text[start : end + 1])).model_dump()


def _judge_messages(
    explanation: dict[str, Any],
    checks: dict[str, object],
    working_memory: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, str]]:
    candidate = dict(explanation)
    if checks.get("geometry_latex_available"):
        candidate["raw_equation"] = (
            "[Lossy PDF extraction omitted; use display_latex and layout_evidence.]"
        )
        candidate["parse"] = {
            "status": "geometry_candidate_available",
            "note": "Compare the selected transcription with the independent geometry candidate.",
        }
    elif checks.get("vision_latex_used"):
        candidate["layout_evidence"] = {
            "note": "Vision supplied the selected crop transcription; it still requires validation."
        }
        candidate["parse"] = {"status": "vision_candidate_selected"}
    return [
        {
            "role": "system",
            "content": (
                "You are an independent LLM-as-a-judge for research-paper math explanations. Evaluate the "
                "candidate only against the raw equation, nearby paper text, deterministic parse, and stated "
                "symbol provenance. Score correctness, paper grounding, symbol coverage, LaTeX fidelity, and "
                "usefulness from 1 to 5. Penalize unsupported meanings and semantic changes in reconstructed "
                "LaTeX. Compare all transcription_candidates and their structural similarities; no candidate is "
                "correct merely because it came from vision, geometry, or SymPy. The raw PDF text is lossy and "
                "may flatten superscripts and subscripts. Crop provenance shows that vision received a bounded "
                "source image, but self-reported confidence is not proof of correctness. Use PDF geometry as "
                "independent evidence rather than unquestioned ground truth. "
                "Do not call DQ, IUQ, or other labels unsupported merely because raw extraction inserted spaces. "
                "Judge usefulness using the step-by-step explanation, symbol provenance, dimensional analysis, "
                "intuition, implementation view, and cited paper evidence. Generic prose or an empty symbol table "
                "must score poorly. Human judge memory below contains rubric guidance and prior calibration cases, "
                "not facts about this paper. Apply relevant principles consistently, use episodes as examples rather "
                "than templates, and never let memory override paper evidence or deterministic validation. Provide "
                "concise written justifications and actionable issues. Do not rewrite the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate this math explanation:\n"
                f"{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
                f"Deterministic checks:\n{json.dumps(checks, ensure_ascii=False, indent=2)}\n\n"
                "Human-aligned judge working memory:\n"
                f"{json.dumps(working_memory or {'principles': [], 'episodes': []}, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _contradicts_present_latex(text: str, latex: str) -> bool:
    corrections = re.findall(r"`([^`]*(?:_|\^)[^`]*)`", text)
    return any(item.replace(" ", "") in latex.replace(" ", "") for item in corrections)


def _trace_phase(value: str) -> str:
    clean = re.sub(r"[^a-z0-9-]+", "-", value.strip().casefold()).strip("-")
    return clean or "initial"


@harness_workflow("math")
def evaluate_math_explanation(
    workspace: Workspace,
    explanation: dict[str, Any],
    progress: Callable[[str, str], None] | None = None,
    *,
    agent_factory: Any | None = None,
    chat_model: Any | None = None,
    phase: str = "initial",
) -> MathJudgeResult:
    phase_key = _trace_phase(phase)
    working_memory = retrieve_math_judge_memory(explanation)
    with observed_stage(
        f"validate-{phase_key}",
        input_data={
            "displayLatex": explanation.get("display_latex"),
            "steps": explanation.get("steps"),
            "symbols": explanation.get("symbols"),
        },
        metadata={"phase": phase_key},
        as_type="guardrail",
    ) as validation:
        checks = deterministic_math_checks(explanation)
        checks["judge_memory"] = {
            "principles": len(working_memory["principles"]),
            "episodes": len(working_memory["episodes"]),
        }
        validation.update(
            output={
                "valid": bool(
                    checks["required_fields_present"]
                    and checks["latex_present"]
                    and checks["latex_braces_balanced"]
                    and checks["latex_commands_safe"]
                    and checks["inline_math_valid"]
                ),
                "missingFields": checks["missing_fields"],
                "inlineMathIssues": checks["inline_math_issues"],
                "symbolCount": checks["symbol_count"],
            },
            metadata={
                "phase": phase_key,
                "inlineMathValid": checks["inline_math_valid"],
                "requiredFieldsPresent": checks["required_fields_present"],
            },
            level=(
                "DEFAULT"
                if checks["inline_math_valid"] and checks["required_fields_present"]
                else "WARNING"
            ),
        )
    provider = os.getenv(
        "PAPER_READER_JUDGE_PROVIDER", os.getenv("PAPER_READER_AGENT_PROVIDER", "ollama")
    )
    primary = os.getenv("PAPER_READER_JUDGE_MODEL", "qwen2.5:7b")
    generator = str(explanation.get("model") or "").split(":", 1)[-1]
    model_names = [primary]
    if chat_model is None and provider == "ollama":
        model_names.extend(
            item.strip()
            for item in os.getenv("PAPER_READER_JUDGE_FALLBACK_MODELS", "qwen2.5:3b").split(",")
            if item.strip()
        )
    model_names = [name for name in dict.fromkeys(model_names) if name != generator] or [primary]
    use_direct_structured_output = agent_factory is None
    if agent_factory is None:
        from langchain.agents import create_agent

        agent_factory = create_agent

    failures: list[str] = []
    judged: dict[str, Any] | None = None
    used_model = "unavailable"
    output_budget = int(os.getenv("PAPER_READER_JUDGE_MAX_TOKENS", "700"))
    for model_attempt, model_name in enumerate(model_names, start=1):
        current_model = chat_model
        if current_model is None:
            config = RunConfig(
                pdf_path=Path(str(explanation.get("source_pdf") or "paper.pdf")),
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
            if progress:
                progress(
                    "math_judge",
                    f"The independent judge is evaluating the explanation with {model_name}.",
                )
            messages = _judge_messages(explanation, checks, working_memory)
            record_context_usage(
                workspace,
                workflow="math_judge",
                provider=provider,
                model=model_name,
                components={
                    "system prompt": messages[0]["content"],
                    "candidate explanation": messages[1]["content"],
                    "structured response schema": response_schema_text(MathJudgePayload),
                },
                reserved_output_tokens=output_budget,
                metadata={
                    "generatorModel": generator,
                    "phase": phase_key,
                    "judgeMemoryPrinciples": len(working_memory["principles"]),
                    "judgeMemoryEpisodes": len(working_memory["episodes"]),
                },
            )
            judge_name = "judge-initial" if phase_key == "initial" else f"judge-after-{phase_key}"
            if model_attempt > 1:
                judge_name = f"{judge_name}-fallback-{model_attempt}"
            if use_direct_structured_output and hasattr(current_model, "with_structured_output"):
                structured_model = current_model.with_structured_output(
                    MathJudgePayload, method="json_schema"
                )
                judged = _structured_payload(
                    invoke_observed(
                        structured_model,
                        messages,
                        name=judge_name,
                        model=f"{provider}:{model_name}",
                        metadata={
                            "generatorModel": generator,
                            "phase": phase_key,
                            "modelAttempt": model_attempt,
                        },
                        as_type="evaluator",
                    )
                )
                used_model = f"{provider}:{model_name}"
                break
            judge = create_harness_agent(
                agent_factory,
                model=current_model,
                tools=[],
                system_prompt=messages[0]["content"],
                name="paper_math_judge",
                response_format=MathJudgePayload,
            )
            result = invoke_observed(
                judge,
                {"messages": [messages[1]]},
                name=judge_name,
                model=f"{provider}:{model_name}",
                metadata={
                    "generatorModel": generator,
                    "phase": phase_key,
                    "modelAttempt": model_attempt,
                },
                as_type="evaluator",
            )
            judged = _structured_payload(result)
            used_model = f"{provider}:{model_name}"
            break
        except HarnessLimitExceeded:
            raise
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")
        if chat_model is not None:
            break

    if judged is None:
        dimensions = {
            name: asdict(JudgeDimension(0.0, "Judge unavailable."))
            for name in [
                "correctness",
                "paper_grounding",
                "symbol_coverage",
                "latex_fidelity",
                "usefulness",
            ]
        }
        return MathJudgeResult(
            verdict="review",
            overall_score=0.0,
            dimensions=dimensions,
            summary="The LLM judge was unavailable; deterministic checks are shown instead.",
            issues=failures,
            deterministic_checks=checks,
            judge_model=used_model,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    dimension_names = [
        "correctness",
        "paper_grounding",
        "symbol_coverage",
        "latex_fidelity",
        "usefulness",
    ]
    dimensions = {
        name: {
            "score": float(judged[name]["score"]),
            "justification": str(judged[name]["justification"]),
        }
        for name in dimension_names
    }
    if checks.get("geometry_latex_matches"):
        latex = str(explanation.get("display_latex") or "")
        dimensions["latex_fidelity"] = {
            "score": 5.0,
            "justification": "Display LaTeX exactly matches the deterministic reconstruction from PDF glyph geometry.",
        }
        replacements = {
            "correctness": "The reconstructed notation and operations agree with the supplied PDF geometry and paper context.",
            "symbol_coverage": "The displayed equation preserves the symbols and nested scripts visible in the PDF geometry.",
        }
        for name, replacement in replacements.items():
            if _contradicts_present_latex(str(dimensions[name]["justification"]), latex):
                dimensions[name]["justification"] = replacement
    scores = [float(dimensions[name]["score"]) for name in dimension_names]
    overall = round(sum(scores) / len(scores), 2)
    hard_failure = (
        not checks["required_fields_present"]
        or not checks["latex_present"]
        or not checks["latex_braces_balanced"]
        or not checks["latex_commands_safe"]
        or not checks["inline_math_valid"]
        or not checks["transcription_confident"]
        or not checks["vision_crop_grounded"]
        or min(scores) < 2
    )
    verdict = (
        "fail" if hard_failure else ("pass" if overall >= 4 and min(scores) >= 3 else "review")
    )
    issues = [str(item) for item in judged.get("issues", [])]
    if checks.get("geometry_latex_matches"):
        latex = str(explanation.get("display_latex") or "")
        issues = [item for item in issues if not _contradicts_present_latex(item, latex)]
    if checks.get("vision_latex_used"):
        issues = [item for item in issues if "geometry_latex" not in item]
    if checks["missing_fields"]:
        issues.append(f"Missing required fields: {', '.join(checks['missing_fields'])}")
    if not checks["latex_present"]:
        issues.append("No display LaTeX was produced.")
    if not checks["transcription_confident"]:
        issues.append(
            "The equation transcription is not independently reliable enough to accept; verify the source crop."
        )
        issues.extend(str(item) for item in checks["transcription_warnings"])
    if not checks["vision_crop_grounded"]:
        issues.append(
            "Vision LaTeX was not linked to a verifiable bounded crop from the source PDF."
        )
    issues.extend(str(item) for item in checks["inline_math_issues"])
    issues = list(dict.fromkeys(issues))
    return MathJudgeResult(
        verdict=verdict,
        overall_score=overall,
        dimensions=dimensions,
        summary=str(judged.get("summary") or ""),
        issues=issues,
        deterministic_checks=checks,
        judge_model=used_model,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )

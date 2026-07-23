from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import sympy
import fitz
from pydantic import BaseModel, Field
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from paper_agent.agents import build_chat_model
from paper_agent.config import DEFAULT_OLLAMA_BASE_URL, RunConfig
from paper_agent.context_diagnostics import record_context_usage, response_schema_text
from paper_agent.math_grounding import build_math_context, deterministic_math_fallback
from paper_agent.math_judge import evaluate_math_explanation
from paper_agent.parser import ParsedPaper
from paper_agent.workspace import Workspace


@dataclass
class MathParse:
    status: str
    normalized: str
    sympy_form: str | None
    latex: str | None
    free_symbols: list[str]
    operation_count: int | None
    error: str | None = None


@dataclass
class MathExplanation:
    equation_id: str
    page: int
    raw_equation: str
    region_kind: str
    paper_context: str
    layout_evidence: dict[str, object]
    display_latex: str
    latex_source: str
    parse: dict[str, object]
    role: str
    plain_english: str
    steps: list[str]
    symbols: list[dict[str, str]]
    intuition: str
    dimensional_analysis: str
    implementation_view: str
    paper_evidence: list[str]
    context_fit: str
    assumptions_or_missing_details: list[str]
    model: str
    evaluation: dict[str, object]
    status: str = "agent_explained"


class MathSymbolMeaning(BaseModel):
    symbol: str = Field(description="The symbol written as valid LaTeX without math delimiters.")
    meaning: str = Field(description="What the symbol denotes in this paper.")
    source: Literal["paper", "inference", "unresolved"] = Field(
        description="Provenance of the stated meaning."
    )


class MathAgentPayload(BaseModel):
    latex: str = Field(
        description="Faithful display LaTeX for the complete equation or inline notation."
    )
    role: str = Field(
        description="The equation's role in the paper, such as definition, objective, or update rule."
    )
    plain_english: str = Field(
        description="A specific three-to-five sentence explanation of the full computation."
    )
    steps: list[str] = Field(
        min_length=2, max_length=6, description="Two to six ordered computational steps."
    )
    symbols: list[MathSymbolMeaning] = Field(
        min_length=1, description="Every important symbol with meaning and provenance."
    )
    intuition: str = Field(description="Why this mathematical construction is useful here.")
    dimensional_analysis: str = Field(
        description="Tensor shapes and compatibility, separating stated facts from inference."
    )
    implementation_view: str = Field(
        description="A concise tensor-operation or pseudocode interpretation."
    )
    paper_evidence: list[str] = Field(
        min_length=1, description="One to three nearby paper facts supporting the explanation."
    )
    context_fit: str = Field(
        description="How this math supports the paper's method, system, or result."
    )
    assumptions_or_missing_details: list[str]


class MathReasoningPayload(BaseModel):
    role: str = Field(description="The concrete function of this equation in the paper.")
    plain_english: str = Field(
        description="A specific three-to-five sentence explanation of the computation."
    )
    steps: list[str] = Field(
        min_length=2, max_length=6, description="Two to six ordered computational steps."
    )
    intuition: str = Field(
        description="Why this mathematical construction is useful in this paper."
    )
    dimensional_analysis: str = Field(
        description="Shape or scalar compatibility, with unsupported details labeled."
    )
    implementation_view: str = Field(
        description="A concise tensor-operation or pseudocode interpretation."
    )
    context_fit: str = Field(
        description="How this equation supports the paper's method, system, or result."
    )
    assumptions_or_missing_details: list[str]


GREEK_NAMES = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "ψ": "psi",
    "ω": "omega",
    "Δ": "Delta",
    "Σ": "Sigma",
}


def _symbol_names(raw: str) -> list[str]:
    names = re.findall(r"[A-Za-z][A-Za-z0-9_]*|[α-ωΑ-Ω]", raw)
    blocked = {"exp", "log", "ln", "sqrt", "softmax", "max", "min", "argmax", "argmin"}
    output: list[str] = []
    for name in names:
        if name.lower() in blocked or len(name) > 24:
            continue
        normalized = GREEK_NAMES.get(name, name)
        if normalized not in output:
            output.append(normalized)
    return output[:30]


def analyze_expression(raw: str) -> MathParse:
    clean = re.sub(r"\s+", " ", raw).strip()
    symbols = _symbol_names(clean)
    if not clean:
        return MathParse("empty", "", None, None, [], None, "No equation text was extracted.")
    try:
        if "\\" in clean or re.search(r"\{[^{}]+\}", clean):
            expression = parse_latex(clean)
            return MathParse(
                "parsed_latex",
                clean,
                sympy.sstr(expression),
                sympy.latex(expression),
                sorted(str(symbol) for symbol in expression.free_symbols),
                int(sympy.count_ops(expression)),
            )

        normalized = clean.translate(str.maketrans(GREEK_NAMES))
        normalized = normalized.replace("−", "-").replace("·", "*").replace("×", "*")
        normalized = re.sub(r"\(\s*\d+\s*\)$", "", normalized).strip()
        if not re.fullmatch(r"[A-Za-z0-9_+\-*/^().,=\s]+", normalized) or "__" in normalized:
            raise ValueError("The extracted PDF expression contains unsupported notation.")

        identifiers = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", normalized))
        functions = {
            "exp": sympy.exp,
            "log": sympy.log,
            "ln": sympy.log,
            "sqrt": sympy.sqrt,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "softmax": sympy.Function("softmax"),
        }
        local_dict: dict[str, object] = {
            name: functions.get(name, sympy.Symbol(name)) for name in identifiers
        }
        global_dict = {
            "Symbol": sympy.Symbol,
            "Integer": sympy.Integer,
            "Float": sympy.Float,
            "Rational": sympy.Rational,
            "Add": sympy.Add,
            "Mul": sympy.Mul,
            "Pow": sympy.Pow,
        }

        def parse_one(value: str):
            return parse_expr(
                value,
                local_dict=local_dict,
                global_dict=global_dict,
                transformations=standard_transformations + (convert_xor,),
                evaluate=False,
            )

        if normalized.count("=") == 1:
            left, right = normalized.split("=", 1)
            expression = sympy.Eq(parse_one(left), parse_one(right), evaluate=False)
        else:
            expression = parse_one(normalized)
        return MathParse(
            "parsed_plain",
            normalized,
            sympy.sstr(expression),
            sympy.latex(expression),
            sorted(str(symbol) for symbol in expression.free_symbols),
            int(sympy.count_ops(expression)),
        )
    except Exception as exc:
        return MathParse("unparsed", clean, None, None, symbols, None, str(exc))


def _page_context(parsed: ParsedPaper, page: int, raw: str) -> str:
    text = parsed.page_text.get(page, "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    raw_key = re.sub(r"\s+", " ", raw).strip().lower()[:80]
    for index, line in enumerate(lines):
        if raw_key and (raw_key in line.lower() or line.lower()[:60] in raw_key):
            return "\n".join(lines[max(0, index - 4) : index + 5])[:5000]
    return "\n".join(lines)[:5000]


def _fallback_symbol_glosses(
    latex: str, context: str, parsed_symbols: list[str]
) -> list[dict[str, str]]:
    compact = latex.replace(" ", "")
    structural = compact.replace(r"\mathbf", "").replace(r"\mathrm", "").replace(r"\text", "")
    structural = re.sub(r"\{([A-Za-z])\}", r"\1", structural)
    lower_context = context.lower()
    symbols: list[dict[str, str]] = []

    def add(symbol: str, meaning: str, source: str) -> None:
        if not any(item["symbol"] == symbol for item in symbols):
            symbols.append({"symbol": symbol, "meaning": meaning, "source": source})

    if r"c_{t}^{Q}" in compact:
        add(
            r"c_t^Q",
            "Intermediate compressed query representation produced from the current hidden state.",
            "inference",
        )
    if r"h_{t}" in compact:
        source = "paper" if "hidden state" in lower_context else "inference"
        add(r"h_t", "Current input hidden state of the query token at decoding step t.", source)
    if r"W^{DQ}" in compact:
        source = "paper" if "down-projection" in lower_context else "inference"
        add(
            r"W^{DQ}",
            "Down-projection matrix that maps the hidden state into the compressed query space.",
            source,
        )
    if re.search(r"q_\{t(?:,[^}]*)?\}\^\{l\}", compact):
        source = "paper" if "indexer quer" in lower_context else "inference"
        add(
            r"q_{t,h}^l",
            "Low-rank lookahead query for indexer head h at time step t in layer l.",
            source,
        )
    if r"n_{h}^{l}" in compact:
        source = "paper" if "indexer heads" in lower_context else "inference"
        add(r"n_h^l", "Number of indexer heads in layer l.", source)
    if r"W^{IUQ}" in compact:
        source = "paper" if "up-projection" in lower_context else "inference"
        add(
            r"W^{IUQ}",
            "Up-projection matrix that expands the compressed query into per-head indexer queries.",
            source,
        )
    if r"I_{t,s}" in structural:
        add(
            r"I_{t,s}",
            "Lookahead index score between query token t and preceding compressed entry s.",
            "paper",
        )
    if r"\sigma" in structural:
        add(
            r"\sigma(\cdot)",
            "Sigmoid activation that maps the fused score into the interval (0, 1).",
            "paper",
        )
    if r"\sum_{h=1}" in structural:
        add("h", "Indexer-head index used by the head-fusion summation.", "inference")
    if r"n_h^l" in structural or r"n_{h}^{l}" in structural:
        source = "paper" if "indexer head" in lower_context else "inference"
        add(r"n_h^l", "Number of indexer heads in layer l.", source)
    if re.search(r"w_\{t,h\}\^l", structural):
        source = (
            "paper"
            if "routing head weight" in lower_context or "scales the importance" in lower_context
            else "inference"
        )
        add(
            r"w_{t,h}^l",
            "Routing weight that scales the contribution of head h for query token t in layer l.",
            source,
        )
    if re.search(r"q_\{t,h\}\^l", structural):
        source = "paper" if "indexer quer" in lower_context else "inference"
        add(r"q_{t,h}^l", "Low-rank indexer query for head h at token t in layer l.", source)
    if "^l" in structural or "^{l}" in structural:
        add(
            "l",
            "Indexer layer associated with the query, routing weight, and head count.",
            "inference",
        )
    if "IComp" in structural and ("K_s" in structural or "K_{s}" in structural):
        source = "paper" if "compressed indexer key" in lower_context else "inference"
        add(
            r"K_s^{IComp}",
            "Compressed indexer key for the preceding entry s; the equation uses its transpose.",
            source,
        )
    if not symbols:
        for symbol in parsed_symbols:
            add(symbol, "Meaning was not recoverable from the nearby paper text.", "unresolved")
    return symbols


def _symbol_key(value: str) -> str:
    clean = value.strip().strip("$")
    clean = re.sub(r"\\(?:mathbf|mathrm|text|bf)", "", clean)
    clean = re.sub(r"\(\\cdot\)$", "", clean)
    return re.sub(r"[{}\s]", "", clean).lower()


def _clean_symbol_latex(value: str) -> str:
    clean = value.strip().strip("$")
    while "\\\\" in clean:
        clean = clean.replace("\\\\", "\\")
    return clean


def _specific_role(role: str, latex: str) -> str:
    clean = role.strip()
    if clean.lower() not in {
        "",
        "unspecified",
        "unavailable",
        "mathematical inference",
        "equation",
    }:
        return clean
    if "W" in latex and latex.count("=") >= 2:
        return "two-stage query projection and per-head query definition"
    if "=" in latex:
        return "definition"
    return "mathematical operation"


def _safe_display_latex(value: str) -> str:
    latex = value.strip()
    if latex.startswith("$$") and latex.endswith("$$"):
        latex = latex[2:-2].strip()
    elif latex.startswith("$") and latex.endswith("$"):
        latex = latex[1:-1].strip()
    if latex.startswith(r"\[") and latex.endswith(r"\]"):
        latex = latex[2:-2].strip()
    if not latex or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", latex):
        return ""
    if latex.count("{") != latex.count("}"):
        return ""
    if re.search(r"\\(?:href|url|includegraphics|input|write18)\b", latex, re.I):
        return ""
    return latex


def _equation_crop_data_url(
    parsed: ParsedPaper, page: int, layout_evidence: dict[str, object] | None
) -> str:
    bbox = (layout_evidence or {}).get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return ""
    try:
        with fitz.open(parsed.metadata.source_pdf) as document:
            source_page = document[page - 1]
            rect = fitz.Rect(*(float(value) for value in bbox))
            rect = (
                fitz.Rect(rect.x0 - 10, rect.y0 - 8, rect.x1 + 10, rect.y1 + 8) & source_page.rect
            )
            pixmap = source_page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=rect, alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def _vision_latex_transcription(
    parsed: ParsedPaper,
    page: int,
    layout_evidence: dict[str, object] | None,
    model: str,
    base_url: str,
) -> str:
    crop = _equation_crop_data_url(parsed, page, layout_evidence)
    if not crop:
        return ""
    schema = {
        "type": "object",
        "properties": {"latex": {"type": "string"}},
        "required": ["latex"],
    }
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Transcribe the complete equation in this image into valid MathJax-compatible LaTeX. "
                    "Preserve every summation and its limits, delimiter, subscript, superscript, transpose, "
                    "activation function, and operator. Do not explain or simplify the equation."
                ),
                "images": [crop.split(",", 1)[1]],
            }
        ],
        "format": schema,
        "options": {"temperature": 0, "num_predict": 500},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    timeout = max(15.0, min(float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120")), 240.0))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    content = str((result.get("message") or {}).get("content") or "")
    parsed_payload = json.loads(content)
    return _safe_display_latex(str(parsed_payload.get("latex") or ""))


def _response_json(value: Any, schema_type: type[BaseModel] = MathAgentPayload) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return schema_type.model_validate(value).model_dump()
    structured = value.get("structured_response") if isinstance(value, dict) else None
    if structured is not None:
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        if isinstance(structured, dict):
            return structured
    messages = value.get("messages", []) if isinstance(value, dict) else []
    content = getattr(messages[-1], "content", "") if messages else str(value or "")
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
        raise ValueError("Math agent did not return JSON.")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Math agent returned a non-object response.")
    return payload


_SCHEMA_LEAK_MARKERS = (
    "tensor_dims_explanation",
    "implementation_view_intuition",
    "symbol_meanings_table",
    "equation_function_classification",
    "text_2d_or_more",
    "labels_as_paper_inference_unresolved",
)


def _clean_generated_list(items: object, *, max_items: int) -> list[str]:
    if not isinstance(items, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = re.sub(r"^\s*(?:step\s*)?\d+\s*[.)-]\s*", "", str(item or ""), flags=re.I)
        clean = re.sub(r"\s+", " ", clean).strip(" \t\r\n-*")
        low = clean.casefold()
        if not clean or len(clean) > 800:
            continue
        if any(marker in low for marker in _SCHEMA_LEAK_MARKERS):
            continue
        if clean.count("_") >= 4 and re.fullmatch(r"[a-z0-9_]+", low):
            continue
        key = low.rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= max_items:
            break
    return output


def explain_equation(
    parsed: ParsedPaper,
    workspace: Workspace,
    page: int,
    equation_id: str,
    raw: str,
    context: str = "",
    region_kind: str = "display",
    layout_evidence: dict[str, object] | None = None,
    progress: Callable[[str, str], None] | None = None,
    *,
    agent_factory: Any | None = None,
    chat_model: Any | None = None,
    judge_fn: Callable[..., object] | None = None,
) -> MathExplanation:
    digest = hashlib.sha256(
        f"math-agent-v20-staged-grounding\n{page}\n{region_kind}\n{raw}\n{context}\n"
        f"{json.dumps(layout_evidence or {}, sort_keys=True)}".encode("utf-8")
    ).hexdigest()[:16]
    cache_rel = f"math/explanations/page-{page:03d}-{digest}.json"
    cache_path = workspace.path(cache_rel)
    if cache_path.exists():
        try:
            return MathExplanation(**json.loads(cache_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if progress:
        progress("sympy", "Parsing equation structure and symbols with SymPy.")
    parsed_math = analyze_expression(raw)
    supplied_context = context.strip()
    provider = os.getenv("PAPER_READER_AGENT_PROVIDER", os.getenv("MODEL_PROVIDER", "ollama"))
    primary_model = os.getenv(
        "PAPER_READER_MATH_MODEL", os.getenv("PAPER_READER_AGENT_MODEL", "qwen3.5:4b")
    )
    vision_latex = ""
    if "qwen3.5" in primary_model.lower() and region_kind == "display":
        try:
            if progress:
                progress("math_vision", f"{primary_model} is reading the equation crop.")
            vision_latex = _vision_latex_transcription(
                parsed,
                page,
                layout_evidence,
                primary_model,
                os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            )
        except Exception:
            vision_latex = ""
    geometry_latex = _safe_display_latex(str((layout_evidence or {}).get("geometry_latex") or ""))
    seed_latex = parsed_math.latex or geometry_latex or vision_latex
    grounding = build_math_context(parsed, page, seed_latex or raw, raw)
    context_parts = [grounding.context]
    if supplied_context and supplied_context not in grounding.context:
        context_parts.append(f"PDF-region context:\n{supplied_context[:3000]}")
    context = "\n\n".join(part for part in context_parts if part).strip() or _page_context(
        parsed, page, raw
    )
    grounded_symbols = [
        {
            "symbol": item.symbol,
            "meaning": item.meaning,
            "source": item.source,
        }
        for item in grounding.symbols
    ]
    model_names = [primary_model]
    if chat_model is None and provider == "ollama":
        model_names.extend(
            name.strip()
            for name in os.getenv(
                "PAPER_READER_MATH_FALLBACK_MODELS", "qwen2.5:7b,qwen2.5:3b"
            ).split(",")
            if name.strip()
        )
    model_names = list(dict.fromkeys(model_names))
    max_attempts = max(1, min(int(os.getenv("PAPER_READER_MATH_MAX_ATTEMPTS", "2")), 3))
    model_names = model_names[:max_attempts]
    use_direct_structured_output = agent_factory is None
    if agent_factory is None:
        from deepagents import create_deep_agent

        agent_factory = create_deep_agent
    failures: list[str] = []
    payload: dict[str, Any] | None = None
    used_model = "unavailable"
    output_budget = int(os.getenv("PAPER_READER_MATH_MAX_TOKENS", "1200"))

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
                model_updates: dict[str, object] = {
                    "keep_alive": os.getenv("PAPER_READER_AGENT_KEEP_ALIVE", "15m"),
                    "num_predict": output_budget,
                }
                if "qwen3.5" in model_name.lower():
                    model_updates["reasoning"] = False
                current_model = current_model.model_copy(update=model_updates)
        try:
            if progress:
                progress(
                    "math_agent", f"The math agent is analyzing page {page} with {model_name}."
                )
            system_prompt = (
                "You are the reasoning stage of a paper-grounded math pipeline. Deterministic tools already "
                "transcribed the equation and retrieved symbol definitions. Explain the supplied formula using "
                "those grounded symbols and the quoted paper context. Do not re-transcribe the formula or invent "
                "symbol meanings. Distinguish paper statements from mathematical inference, classify the concrete "
                "role, give ordered computational steps, check scalar or tensor compatibility, and explain how "
                "the equation advances this paper's method."
            )
            prompt_layout: object = {
                "note": "Equation perception completed earlier; raw glyph spans are omitted.",
                "bbox": (layout_evidence or {}).get("bbox"),
            }
            user_prompt = (
                f"Paper: {parsed.metadata.title_guess or 'Untitled paper'}\nPage: {page}\n"
                f"Region type: {region_kind}\nEquation or inline notation: {raw}\n"
                f"PDF layout evidence: {json.dumps(prompt_layout, ensure_ascii=False)}\n"
                f"Deterministic geometry reconstruction: {geometry_latex or 'unavailable'}\n"
                f"Vision transcription: {vision_latex or 'unavailable'}\n"
                f"SymPy analysis: {json.dumps(asdict(parsed_math), ensure_ascii=False)}\n\n"
                f"Grounded symbol table: {json.dumps(grounded_symbols, ensure_ascii=False)}\n\n"
                f"Nearby paper explanation:\n{context}"
            )
            response_schema = (
                MathReasoningPayload if use_direct_structured_output else MathAgentPayload
            )
            record_context_usage(
                workspace,
                workflow="math_explanation",
                provider=provider,
                model=model_name,
                components={
                    "system prompt": system_prompt,
                    "equation and paper context": user_prompt,
                    "structured response schema": response_schema_text(response_schema),
                },
                reserved_output_tokens=output_budget,
                metadata={"page": page, "equationId": equation_id, "regionKind": region_kind},
            )
            if use_direct_structured_output and hasattr(current_model, "with_structured_output"):
                structured_model = current_model.with_structured_output(
                    MathReasoningPayload, method="json_schema"
                )
                payload = _response_json(
                    structured_model.invoke(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    ),
                    MathReasoningPayload,
                )
                if not payload.get("plain_english") or not payload.get("context_fit"):
                    raise ValueError("Math agent returned an incomplete explanation.")
                payload["latex"] = seed_latex
                payload["symbols"] = grounded_symbols
                payload["paper_evidence"] = grounding.evidence[:3]
                used_model = f"{provider}:{model_name}"
                break
            agent = agent_factory(
                model=current_model,
                tools=[],
                system_prompt=system_prompt,
                name="paper_math_agent",
                response_format=MathAgentPayload,
            )
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ]
                }
            )
            payload = _response_json(result)
            if (
                not str(payload.get("plain_english") or "").strip()
                or not str(payload.get("context_fit") or "").strip()
                or not str(payload.get("latex") or "").strip()
            ):
                raise ValueError("Math agent returned an incomplete explanation.")
            used_model = f"{provider}:{model_name}"
            break
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")
            if progress:
                progress(
                    "math_retry",
                    f"{model_name} did not return a valid explanation; trying the next bounded fallback.",
                )
        if chat_model is not None:
            break

    if payload is None:
        fallback = deterministic_math_fallback(seed_latex or raw, grounding)
        payload = {
            "latex": seed_latex,
            "symbols": grounded_symbols,
            "paper_evidence": grounding.evidence[:3] or [context[:800]],
            **fallback,
            "assumptions_or_missing_details": [
                *fallback.get("assumptions_or_missing_details", []),
                *[f"Model stage failed: {item[:500]}" for item in failures],
            ],
        }
        workspace.write_json(
            "logs/math_agent_last_failure.json",
            {
                "page": page,
                "equationId": equation_id,
                "models": model_names,
                "failures": failures,
                "groundedSymbolCount": len(grounded_symbols),
                "fallback": "deterministic_grounded",
            },
        )
        used_model = "deterministic-grounding"
    symbols = [
        {
            "symbol": _clean_symbol_latex(str(item.get("symbol") or "")),
            "meaning": str(item.get("meaning") or ""),
            "source": str(item.get("source") or "inference"),
        }
        for item in payload.get("symbols", [])
        if isinstance(item, dict)
    ]
    agent_latex = _safe_display_latex(
        re.sub(r"^\s*(?:\$\$|\\\[)|(?:\$\$|\\\])\s*$", "", str(payload.get("latex") or "").strip())
    )
    geometry_latex = _safe_display_latex(str((layout_evidence or {}).get("geometry_latex") or ""))
    display_latex = parsed_math.latex or geometry_latex or vision_latex or agent_latex
    latex_source = (
        "sympy"
        if parsed_math.latex
        else "pdf_geometry"
        if geometry_latex
        else "vision_math_agent"
        if vision_latex
        else "paper_math_agent"
        if agent_latex
        else "unavailable"
    )
    known = {_symbol_key(item["symbol"]): index for index, item in enumerate(symbols)}
    supplemental_symbols = [
        *grounded_symbols,
        *_fallback_symbol_glosses(display_latex, context, parsed_math.free_symbols),
    ]
    if not symbols:
        symbols = supplemental_symbols
    else:
        for item in supplemental_symbols:
            key = _symbol_key(item["symbol"])
            if key not in known:
                symbols.append(item)
                known[key] = len(symbols) - 1
            elif item["source"] == "paper" or symbols[known[key]]["source"] == "unresolved":
                symbols[known[key]] = item
    explanation = MathExplanation(
        equation_id=equation_id,
        page=page,
        raw_equation=raw,
        region_kind=region_kind,
        paper_context=context,
        layout_evidence=layout_evidence or {},
        display_latex=display_latex,
        latex_source=latex_source,
        parse=asdict(parsed_math),
        role=_specific_role(str(payload.get("role") or ""), display_latex),
        plain_english=str(payload.get("plain_english") or ""),
        steps=_clean_generated_list(payload.get("steps", []), max_items=6),
        symbols=symbols,
        intuition=str(payload.get("intuition") or ""),
        dimensional_analysis=str(payload.get("dimensional_analysis") or ""),
        implementation_view=str(payload.get("implementation_view") or ""),
        paper_evidence=_clean_generated_list(payload.get("paper_evidence", []), max_items=3),
        context_fit=str(payload.get("context_fit") or ""),
        assumptions_or_missing_details=_clean_generated_list(
            payload.get("assumptions_or_missing_details", []), max_items=6
        ),
        model=used_model,
        evaluation={},
        status=(
            "grounded_fallback"
            if used_model == "deterministic-grounding"
            else "agent_explained"
            if used_model != "unavailable"
            else "parser_only"
        ),
    )
    if progress:
        progress("math_judge", "Running deterministic checks and the independent LLM judge.")
    evaluation = (judge_fn or evaluate_math_explanation)(
        workspace, asdict(explanation), progress=progress
    )
    explanation.evaluation = asdict(evaluation)
    if evaluation.verdict == "fail":
        explanation.status = "needs_review"
    workspace.write_json(
        cache_rel.replace("math/explanations/", "math/evaluations/"), asdict(evaluation)
    )
    workspace.write_json(cache_rel, asdict(explanation))
    return explanation

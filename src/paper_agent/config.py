from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PROVIDER = "ollama"  # openai | ollama
DEFAULT_OPENAI_MODEL = "openai:gpt-5.5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OUTPUT_DIR = Path("paper_report")
DEFAULT_MAX_CHARS = 120_000
DEFAULT_TAVILY_MAX_RESULTS = 5

VALID_MODES = {"paper-only", "research"}
VALID_PROVIDERS = {"openai", "ollama"}


@dataclass(frozen=True)
class RunConfig:
    pdf_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    mode: str = "paper-only"
    model_provider: str = DEFAULT_MODEL_PROVIDER
    model: str = DEFAULT_OLLAMA_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    max_chars: int = DEFAULT_MAX_CHARS
    use_tavily: bool = False
    save_workspace: bool = True
    strict_json: bool = True
    no_agent: bool = False
    tavily_max_results: int = DEFAULT_TAVILY_MAX_RESULTS
    clean_output: bool = True
    run_evaluator: bool = True
    auto_repair_attempts: int = 1

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {self.mode}. Use one of {sorted(VALID_MODES)}")
        if self.model_provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {self.model_provider}. Use one of {sorted(VALID_PROVIDERS)}"
            )
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.tavily_max_results <= 0:
            raise ValueError("tavily_max_results must be positive")
        if self.auto_repair_attempts < 0:
            raise ValueError("auto_repair_attempts must be >= 0")

    @property
    def research_enabled(self) -> bool:
        return self.mode == "research" or self.use_tavily

    @property
    def using_ollama(self) -> bool:
        return self.model_provider.lower() == "ollama"

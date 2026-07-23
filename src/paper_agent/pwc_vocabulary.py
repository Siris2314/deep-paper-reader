from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "agent_memory" / "pwc_methods_vocabulary.json"
PWC_METHODS_PARQUET_URL = (
    "https://huggingface.co/datasets/pwc-archive/methods/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
PWC_METHODS_DATASET_URL = "https://huggingface.co/datasets/pwc-archive/methods"
PWC_SNAPSHOT_DATE = "2025-07-28"
PWC_LICENSE = "CC-BY-SA-4.0"
_CACHE_VERSION = 1
_VOCABULARY_LOCK = threading.RLock()
_MEMORY_CACHE: dict[Path, tuple[int, list["VocabularyTerm"]]] = {}


@dataclass(frozen=True)
class VocabularyTerm:
    term: str
    full_name: str
    num_papers: int
    source: str = "papers_with_code_archive"


def vocabulary_cache_path() -> Path:
    configured = os.getenv("PWC_VOCAB_CACHE_PATH")
    if not configured:
        return DEFAULT_CACHE_PATH
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _clean_term(value: object) -> str:
    term = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n.,;:")
    if not 3 <= len(term) <= 120 or not re.search(r"[A-Za-z]", term):
        return ""
    if term.lower().startswith(("http://", "https://")):
        return ""
    return term


def _read_cache(path: Path) -> list[VocabularyTerm] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != _CACHE_VERSION or not isinstance(payload.get("terms"), list):
        return None
    output: list[VocabularyTerm] = []
    for item in payload["terms"]:
        if not isinstance(item, dict):
            continue
        try:
            output.append(VocabularyTerm(**item))
        except (TypeError, ValueError):
            continue
    return output


def _download_file(url: str, target: Path, timeout: float = 30.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "deep-paper-agent/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def _build_vocabulary(parquet_path: Path) -> list[VocabularyTerm]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Papers with Code vocabulary setup requires pyarrow. Install project requirements first."
        ) from exc

    table = pq.read_table(parquet_path, columns=["name", "full_name", "num_papers"])
    by_key: dict[str, VocabularyTerm] = {}
    for row in table.to_pylist():
        name = _clean_term(row.get("name"))
        full_name = _clean_term(row.get("full_name")) or name
        try:
            num_papers = max(0, int(row.get("num_papers") or 0))
        except (TypeError, ValueError):
            num_papers = 0
        for term in (name, full_name):
            if not term:
                continue
            item = VocabularyTerm(term=term, full_name=full_name or term, num_papers=num_papers)
            key = term.casefold()
            if key not in by_key or item.num_papers > by_key[key].num_papers:
                by_key[key] = item
    return sorted(by_key.values(), key=lambda item: (-item.num_papers, item.term.casefold()))


def _write_cache(path: Path, terms: list[VocabularyTerm]) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "source": PWC_METHODS_DATASET_URL,
        "snapshot_date": PWC_SNAPSHOT_DATE,
        "license": PWC_LICENSE,
        "terms": [asdict(item) for item in terms],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_method_vocabulary(
    *,
    cache_path: Path | None = None,
    allow_network: bool | None = None,
    downloader: Callable[[str, Path, float], None] | None = None,
) -> list[VocabularyTerm]:
    """Load the static Papers with Code methods snapshot, downloading it once when needed."""

    if not _env_enabled("PWC_VOCAB_ENABLED", True):
        return []
    target = (cache_path or vocabulary_cache_path()).resolve()
    network_allowed = (
        _env_enabled("PWC_VOCAB_ALLOW_NETWORK", True) if allow_network is None else allow_network
    )

    with _VOCABULARY_LOCK:
        if target.exists():
            fingerprint = target.stat().st_mtime_ns
            memory_hit = _MEMORY_CACHE.get(target)
            if memory_hit and memory_hit[0] == fingerprint:
                return memory_hit[1]
            cached = _read_cache(target)
            if cached is not None:
                _MEMORY_CACHE[target] = (fingerprint, cached)
                return cached
        if not network_allowed:
            return []

        target.parent.mkdir(parents=True, exist_ok=True)
        parquet_path = target.with_suffix(".parquet.part")
        try:
            (downloader or _download_file)(
                PWC_METHODS_PARQUET_URL,
                parquet_path,
                float(os.getenv("PWC_VOCAB_DOWNLOAD_TIMEOUT", "30")),
            )
            terms = _build_vocabulary(parquet_path)
            _write_cache(target, terms)
            _MEMORY_CACHE[target] = (target.stat().st_mtime_ns, terms)
            return terms
        except Exception:
            # Highlighting must still work from paper-local terms when the archive is unavailable.
            return []
        finally:
            parquet_path.unlink(missing_ok=True)


def terms_present_in_text(text: str, vocabulary: list[VocabularyTerm]) -> list[VocabularyTerm]:
    """Return archive terms present in this paper without adding the full catalog as candidates."""

    folded = text.casefold()
    present: list[VocabularyTerm] = []
    for item in vocabulary:
        case_sensitive = item.term.isupper() and any(char.isalpha() for char in item.term)
        haystack = text if case_sensitive else folded
        needle = item.term if case_sensitive else item.term.casefold()
        start = haystack.find(needle)
        while start >= 0:
            end = start + len(needle)
            left_ok = start == 0 or not haystack[start - 1].isalnum()
            right_ok = end == len(haystack) or not haystack[end].isalnum()
            if left_ok and right_ok:
                present.append(item)
                break
            start = haystack.find(needle, start + 1)
    return present

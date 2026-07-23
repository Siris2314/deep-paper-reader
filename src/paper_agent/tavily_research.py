from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TavilyPrefetchResult:
    enabled: bool
    attempted: bool
    success: bool
    reason: str
    queries: list[str]
    result_count: int


def _safe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return obj


def _normalize_search_response(query: str, response: Any) -> dict[str, Any]:
    payload = _safe_model_dump(response)
    if not isinstance(payload, dict):
        payload = {"raw": str(payload)}

    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []

    normalized_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            item = _safe_model_dump(item)
        if not isinstance(item, dict):
            item = {"raw": str(item)}
        normalized_results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content") or item.get("snippet"),
                "score": item.get("score"),
                "raw": item,
            }
        )

    return {
        "query": query,
        "answer": payload.get("answer"),
        "results": normalized_results,
        "raw": payload,
    }


def build_tavily_tools(enabled: bool) -> list[Any]:
    """Build Tavily tools lazily for Deep Agent optional follow-up search."""
    if not enabled or not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        from langchain_tavily import TavilyExtract, TavilySearch
    except Exception:
        return []

    tools: list[Any] = [
        TavilySearch(
            max_results=int(os.getenv("TAVILY_MAX_RESULTS", "8")),
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
            include_answer=True,
            include_raw_content=False,
        ),
        TavilyExtract(),
    ]

    # Optional newer Tavily tools. Fail closed if installed version does not expose them.
    try:
        from langchain_tavily import TavilyResearch  # type: ignore

        tools.append(TavilyResearch(max_results=int(os.getenv("TAVILY_MAX_RESULTS", "8"))))
    except Exception:
        pass

    try:
        from langchain_tavily import TavilyCrawl, TavilyMap  # type: ignore

        tools.extend([TavilyMap(), TavilyCrawl()])
    except Exception:
        pass

    return tools


def build_research_queries(parsed: Any) -> list[str]:
    """Create deterministic web-search queries from parsed paper metadata/content."""
    metadata_obj = getattr(parsed, "metadata", None)
    if hasattr(metadata_obj, "model_dump"):
        metadata = metadata_obj.model_dump()
    elif isinstance(metadata_obj, dict):
        metadata = metadata_obj
    else:
        metadata = {}
    title = metadata.get("title_guess") or "research paper"

    text = getattr(parsed, "full_text", "") or ""
    arxiv_match = None
    for token in text.replace("[", " ").replace("]", " ").split():
        cleaned = token.strip().strip(",.;()")
        if cleaned.startswith("arXiv:"):
            arxiv_match = cleaned.replace("arXiv:", "")
            break
        if cleaned[:4].isdigit() and "." in cleaned and "v" in cleaned:
            arxiv_match = cleaned
            break

    queries = [
        f'"{title}"',
        f'"{title}" arXiv',
        f'"{title}" GitHub code',
        f'"{title}" Hugging Face model',
        '"Lookahead Sparse Attention" DeepSeek FlashMemory',
        '"FlashMemory-DeepSeek-V4" related work',
        "LongBench-v2 LongMemEval RULER MRCR long context benchmark",
    ]
    if arxiv_match:
        queries.insert(0, f"arXiv {arxiv_match}")

    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            deduped.append(q)
    return deduped


def run_tavily_prefetch(
    parsed: Any, workspace: Any, enabled: bool, max_results: int = 5
) -> TavilyPrefetchResult:
    """Run Tavily deterministically before the agent.

    This is intentionally outside the LLM loop. In v2, Tavily was merely exposed as an optional tool;
    local models often never call it. This function forces research-mode web retrieval and writes the
    results into web/ so the Deep Agent can consume them as context.
    """
    queries = build_research_queries(parsed)
    status_base = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "enabled": enabled,
        "attempted": False,
        "success": False,
        "reason": "not enabled",
        "queries": queries,
        "result_count": 0,
    }

    if not enabled:
        workspace.write_json("web/tavily_status.json", status_base)
        workspace.write_json("web/tavily_queries.json", queries)
        workspace.write_json("web/tavily_sources.json", [])
        return TavilyPrefetchResult(False, False, False, "not enabled", queries, 0)

    if not os.getenv("TAVILY_API_KEY"):
        status_base.update(
            {
                "attempted": False,
                "reason": "TAVILY_API_KEY missing. Put it in .env or set it in the shell.",
            }
        )
        workspace.write_json("web/tavily_status.json", status_base)
        workspace.write_json("web/tavily_queries.json", queries)
        workspace.write_json("web/tavily_sources.json", [])
        return TavilyPrefetchResult(True, False, False, status_base["reason"], queries, 0)

    try:
        from langchain_tavily import TavilySearch
    except Exception as exc:
        reason = f"langchain_tavily import failed: {exc}"
        status_base.update({"attempted": False, "reason": reason})
        workspace.write_json("web/tavily_status.json", status_base)
        workspace.write_json("web/tavily_queries.json", queries)
        workspace.write_json("web/tavily_sources.json", [])
        return TavilyPrefetchResult(True, False, False, reason, queries, 0)

    search = TavilySearch(
        max_results=max_results,
        search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
        include_answer=True,
        include_raw_content=False,
    )

    search_runs: list[dict[str, Any]] = []
    flat_sources: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for query in queries:
        try:
            response = search.invoke({"query": query})
            normalized = _normalize_search_response(query, response)
            search_runs.append(normalized)
            for item in normalized.get("results", []):
                flat_sources.append(
                    {
                        "query": query,
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": item.get("content"),
                        "score": item.get("score"),
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "query": query,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    # URL/title-level dedupe, preserving order.
    deduped_sources: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for source in flat_sources:
        key = str(source.get("url") or source.get("title") or source.get("content") or "").lower()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_sources.append(source)

    success = bool(deduped_sources)
    reason = "ok" if success else "Tavily returned no usable sources"
    if errors and not success:
        reason = f"all Tavily searches failed; first error: {errors[0]['error']}"

    status = dict(status_base)
    status.update(
        {
            "attempted": True,
            "success": success,
            "reason": reason,
            "result_count": len(deduped_sources),
            "error_count": len(errors),
        }
    )

    workspace.write_json("web/tavily_status.json", status)
    workspace.write_json("web/tavily_queries.json", queries)
    workspace.write_json("web/tavily_raw_runs.json", search_runs)
    workspace.write_json("web/tavily_errors.json", errors)
    workspace.write_json("web/tavily_sources.json", deduped_sources)

    summary_lines = [
        "# Tavily Web Research Prefetch",
        "",
        f"Status: {reason}",
        f"Queries: {len(queries)}",
        f"Sources: {len(deduped_sources)}",
        "",
    ]
    for i, source in enumerate(deduped_sources[:25], start=1):
        summary_lines.append(f"## Source {i}")
        summary_lines.append(f"Title: {source.get('title')}")
        summary_lines.append(f"URL: {source.get('url')}")
        summary_lines.append(f"Query: {source.get('query')}")
        content = (source.get("content") or "").replace("\n", " ")
        summary_lines.append(f"Snippet: {content[:700]}")
        summary_lines.append("")
    workspace.write_text("web/tavily_summary.md", "\n".join(summary_lines))

    return TavilyPrefetchResult(True, True, success, reason, queries, len(deduped_sources))

"""Document-bound scholarly identity and explicitly fetched external metadata.

External records are discovery context, never proof of claims made by the uploaded paper.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Any

from paper_agent.arxiv_source import paper_version
from paper_agent.workspace import Workspace

CACHE = "research/scholarly_metadata.json"
MAX_BYTES = 2_000_000
NEGATIVE_TTL = 300
DOI = re.compile(r"(?i)(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)")
ALLOWED_HOSTS = {"api.crossref.org", "api.openalex.org", "api.semanticscholar.org"}
_LOCK = threading.Lock()
_LAST_REQUEST: dict[str, float] = {}


def document_key(parsed) -> str:
    return hashlib.sha256(parsed.full_text.encode("utf-8")).hexdigest()


def _clean_doi(value: str) -> str:
    return value.rstrip(".,;:)]}").casefold()


def paper_identity(parsed) -> dict[str, Any]:
    first_page = parsed.page_text.get(1, "")
    dois = sorted({_clean_doi(match.group(1)) for match in DOI.finditer(first_page)})
    version = paper_version(parsed)
    return {
        "status": "exact_identifier" if len(dois) == 1 or version else "local_only",
        "doi": dois[0] if len(dois) == 1 else None,
        "doi_candidates": dois,
        "arxiv_version": version,
        "arxiv_id": re.sub(r"v\d+$", "", version, flags=re.I) if version else None,
        "title": parsed.metadata.title_guess,
        "authors": parsed.metadata.authors_guess,
        "evidence": "Identifiers extracted from uploaded PDF page 1 only.",
    }


def scholarly_json_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in ALLOWED_HOSTS:
        raise ValueError("Scholarly metadata URL is not allowlisted.")
    host = str(parsed_url.hostname)
    for attempt in range(2):
        with _LOCK:
            delay = max(0.0, 0.35 - (time.monotonic() - _LAST_REQUEST.get(host, 0.0)))
            if delay:
                time.sleep(delay)
            _LAST_REQUEST[host] = time.monotonic()
        request_headers = {"User-Agent": "DeepPaperReader/0.1 scholarly-metadata"}
        request_headers.update(headers or {})
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=request_headers), timeout=15
            ) as response:
                final = urllib.parse.urlparse(response.geturl())
                if final.scheme != "https" or final.hostname != host:
                    raise ValueError("Metadata provider redirected outside its allowlisted host.")
                if "json" not in response.headers.get("Content-Type", "").casefold():
                    raise ValueError("Metadata provider returned a non-JSON response.")
                content = response.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError("Metadata provider response exceeded the size limit.")
            payload = json.loads(content.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except urllib.error.HTTPError as exc:
            if attempt == 0 and exc.code in {429, 500, 502, 503, 504}:
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                try:
                    time.sleep(min(3.0, max(0.25, float(retry_after))))
                except ValueError:
                    time.sleep(0.5)
                continue
            raise
    return {}


def _normalized_title(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _title_agreement(local: str | None, external: str | None) -> float | None:
    if not _normalized_title(local) or not _normalized_title(external):
        return None
    return round(
        SequenceMatcher(None, _normalized_title(local), _normalized_title(external)).ratio(), 3
    )


def _identity_status(agreement: float | None) -> str:
    return "ready" if agreement is None or agreement >= 0.55 else "identity_mismatch"


def _crossref(identity: dict[str, Any]) -> dict[str, Any]:
    doi = identity.get("doi")
    if not doi:
        return {"status": "identifier_unavailable"}
    headers: dict[str, str] = {}
    email = os.getenv("CROSSREF_MAILTO", "").strip()
    if email:
        headers["User-Agent"] = f"DeepPaperReader/0.1 (mailto:{email})"
    payload = scholarly_json_get(
        f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}", headers
    )
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    title_values = message.get("title") if isinstance(message.get("title"), list) else []
    title = str(title_values[0]) if title_values else None
    agreement = _title_agreement(identity.get("title"), title)
    return {
        "status": _identity_status(agreement),
        "title": title,
        "title_agreement": agreement,
        "doi": message.get("DOI") or doi,
        "type": message.get("type"),
        "publisher": message.get("publisher"),
        "published": message.get("published"),
        "authors": [
            " ".join(
                filter(None, [str(item.get("given") or ""), str(item.get("family") or "")])
            ).strip()
            for item in (message.get("author") or [])[:30]
            if isinstance(item, dict)
        ],
        "url": message.get("URL") or f"https://doi.org/{doi}",
        "relation": message.get("relation") if isinstance(message.get("relation"), dict) else {},
    }


def _openalex_headers() -> dict[str, str]:
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _openalex(identity: dict[str, Any]) -> dict[str, Any]:
    doi = identity.get("doi")
    if not doi:
        return {"status": "identifier_unavailable"}
    fields = (
        "id,doi,display_name,publication_year,cited_by_count,type,authorships,"
        "primary_location,open_access,related_works"
    )
    url = (
        "https://api.openalex.org/works/https://doi.org/"
        + urllib.parse.quote(doi, safe="/")
        + "?"
        + urllib.parse.urlencode({"select": fields})
    )
    work = scholarly_json_get(url, _openalex_headers())
    title = str(work.get("display_name") or "") or None
    agreement = _title_agreement(identity.get("title"), title)
    related_ids = [
        str(item).rsplit("/", 1)[-1]
        for item in (work.get("related_works") or [])[:8]
        if re.fullmatch(r"(?:https://openalex\.org/)?W\d+", str(item), re.I)
    ]
    related: list[dict[str, Any]] = []
    if related_ids and _identity_status(agreement) == "ready":
        params = urllib.parse.urlencode(
            {
                "filter": "openalex:" + "|".join(related_ids),
                "per_page": str(len(related_ids)),
                "select": "id,doi,display_name,publication_year,cited_by_count,open_access",
            }
        )
        batch = scholarly_json_get(f"https://api.openalex.org/works?{params}", _openalex_headers())
        for item in (batch.get("results") or [])[:8]:
            if not isinstance(item, dict):
                continue
            related.append(
                {
                    "id": item.get("id"),
                    "doi": item.get("doi"),
                    "title": item.get("display_name"),
                    "year": item.get("publication_year"),
                    "citation_count": item.get("cited_by_count"),
                    "open_access": item.get("open_access"),
                }
            )
    return {
        "status": _identity_status(agreement),
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": title,
        "title_agreement": agreement,
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "citation_count": work.get("cited_by_count"),
        "authors": [
            str(item.get("author", {}).get("display_name") or "")
            for item in (work.get("authorships") or [])[:30]
            if isinstance(item, dict) and isinstance(item.get("author"), dict)
        ],
        "primary_location": work.get("primary_location"),
        "open_access": work.get("open_access"),
        "related_works": related,
    }


def _semantic_scholar(identity: dict[str, Any]) -> dict[str, Any]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", os.getenv("S2_API_KEY", "")).strip()
    allow_shared = os.getenv("SEMANTIC_SCHOLAR_ALLOW_UNAUTHENTICATED", "false").casefold() in {
        "1",
        "true",
        "yes",
    }
    if not key and not allow_shared:
        return {"status": "not_configured"}
    identifier = (
        f"ARXIV:{identity['arxiv_id']}"
        if identity.get("arxiv_id")
        else f"DOI:{identity['doi']}"
        if identity.get("doi")
        else None
    )
    if not identifier:
        return {"status": "identifier_unavailable"}
    fields = (
        "title,year,authors,url,externalIds,citationCount,referenceCount,isOpenAccess,openAccessPdf"
    )
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        + urllib.parse.quote(identifier, safe=":")
        + "?"
        + urllib.parse.urlencode({"fields": fields})
    )
    headers = {"x-api-key": key} if key else {}
    work = scholarly_json_get(url, headers)
    title = str(work.get("title") or "") or None
    agreement = _title_agreement(identity.get("title"), title)
    return {
        "status": _identity_status(agreement),
        "paper_id": work.get("paperId"),
        "title": title,
        "title_agreement": agreement,
        "year": work.get("year"),
        "authors": [
            str(item.get("name") or "")
            for item in (work.get("authors") or [])[:30]
            if isinstance(item, dict)
        ],
        "url": work.get("url"),
        "external_ids": work.get("externalIds"),
        "citation_count": work.get("citationCount"),
        "reference_count": work.get("referenceCount"),
        "is_open_access": work.get("isOpenAccess"),
        "open_access_pdf": work.get("openAccessPdf"),
    }


def local_metadata(parsed) -> dict[str, Any]:
    return {
        "status": "not_loaded",
        "document_key": document_key(parsed),
        "identity": paper_identity(parsed),
        "checked_at": None,
        "providers": {},
    }


def load_metadata(parsed, workspace: Workspace) -> dict[str, Any]:
    try:
        cached = json.loads(workspace.read_text(CACHE))
        if cached.get("document_key") == document_key(parsed):
            return cached
    except (OSError, ValueError, AttributeError):
        pass
    return local_metadata(parsed)


def save_local_identity(parsed, workspace: Workspace) -> dict[str, Any]:
    result = local_metadata(parsed)
    workspace.write_json(CACHE, result)
    return result


def refresh_metadata(parsed, workspace: Workspace) -> dict[str, Any]:
    cached = load_metadata(parsed, workspace)
    if cached["status"] in {"ready", "partial"}:
        return cached
    if cached.get("checked_at") and time.time() - float(cached["checked_at"]) < NEGATIVE_TTL:
        return cached
    identity = paper_identity(parsed)
    providers: dict[str, Any] = {}
    for name, provider in (
        ("crossref", _crossref),
        ("openalex", _openalex),
        ("semantic_scholar", _semantic_scholar),
    ):
        try:
            providers[name] = provider(identity)
        except urllib.error.HTTPError as exc:
            providers[name] = {"status": f"http_{exc.code}"}
        except Exception as exc:
            providers[name] = {"status": "unavailable", "error": str(exc)[:300]}
    ready = sum(item.get("status") == "ready" for item in providers.values())
    result = {
        "status": "ready" if ready >= 2 else "partial" if ready else "unavailable",
        "document_key": document_key(parsed),
        "identity": identity,
        "checked_at": time.time(),
        "providers": providers,
        "warning": "External metadata and related-work edges are discovery context, not evidence for the uploaded paper's claims.",
    }
    workspace.write_json(CACHE, result)
    return result


def retrieve_metadata(
    parsed, workspace: Workspace, query: str, limit: int = 5
) -> list[dict[str, Any]]:
    if not re.search(
        r"\b(?:doi|metadata|author|published|citation|related|similar|open access|prior work)\b",
        query,
        re.I,
    ):
        return []
    cached = load_metadata(parsed, workspace)
    if cached["status"] not in {"ready", "partial"}:
        return []
    results: list[dict[str, Any]] = []
    for name, provider in cached.get("providers", {}).items():
        if provider.get("status") != "ready":
            continue
        url = provider.get("url") or provider.get("id")
        if not url and provider.get("doi"):
            url = "https://doi.org/" + str(provider["doi"]).removeprefix("https://doi.org/")
        results.append(
            {
                "id": f"META:{name}",
                "provider": name,
                "text": json.dumps(provider, ensure_ascii=False),
                "url": url,
                "title": provider.get("title") or f"{name} metadata",
            }
        )
    for index, item in enumerate(
        cached.get("providers", {}).get("openalex", {}).get("related_works", [])
    ):
        results.append(
            {
                "id": f"RELATED:{index + 1}",
                "provider": "openalex",
                "text": json.dumps(item, ensure_ascii=False),
                "url": item.get("id") or item.get("doi"),
                "title": item.get("title") or "Related work",
            }
        )
    return results[: max(1, min(limit, 10))]

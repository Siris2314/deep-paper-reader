"""Explicit, cached scholarly full-text access with document identity provenance."""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from paper_agent.scholarly_metadata import document_key, load_metadata, scholarly_json_get


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Full-text content redirects are not accepted.")


def parse_tei(content: bytes) -> list[dict]:
    if (
        len(content) > 8_000_000
        or b"<!DOCTYPE" in content.upper()
        or b"<!ENTITY" in content.upper()
    ):
        raise ValueError("Unsupported or oversized XML.")
    root = ET.fromstring(content)
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    result = []
    for div in root.findall(".//t:body//t:div", ns):
        heading = " ".join(div.findtext("t:head", "", ns).split())
        for paragraph in div.findall("t:p", ns):
            text = " ".join("".join(paragraph.itertext()).split())
            if len(text) >= 30:
                result.append(
                    {"id": f"OPENALEX:{len(result) + 1}", "text": text[:6000], "section": heading}
                )
    return result[:2000]


def load_fulltext(parsed, workspace) -> dict:
    try:
        data = json.loads(workspace.read_text("research/openalex_fulltext.json"))
        if data.get("document_key") == document_key(parsed):
            return data
    except (OSError, ValueError):
        pass
    return {"status": "not_loaded", "items": []}


def ingest_openalex_fulltext(parsed, workspace) -> dict:
    cached = load_fulltext(parsed, workspace)
    if cached.get("status") == "ready" or time.time() - cached.get("checked_at", 0) < 300:
        return cached
    key = os.getenv("OPENALEX_API_KEY", "").strip()
    metadata = load_metadata(parsed, workspace)
    work = metadata.get("providers", {}).get("openalex", {})
    work_id = str(work.get("id") or "").rsplit("/", 1)[-1]
    if not key or work.get("status") != "ready" or not re.fullmatch(r"W\d+", work_id):
        return {"status": "metadata_or_key_required", "items": []}
    result = {
        "status": "unavailable",
        "document_key": document_key(parsed),
        "work_id": work_id,
        "checked_at": time.time(),
        "items": [],
        "version_alignment": "unverified_external_copy",
    }
    url = f"https://content.openalex.org/works/{work_id}.grobid-xml"
    try:
        request = urllib.request.Request(
            url + "?" + urllib.parse.urlencode({"api_key": key}),
            headers={"Accept-Encoding": "gzip"},
        )
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=25) as response:
            content = response.read(8_000_001)
        if len(content) > 8_000_000:
            raise ValueError("Content size limit")
        if content.startswith(b"\x1f\x8b"):
            import io

            with gzip.GzipFile(fileobj=io.BytesIO(content)) as stream:
                content = stream.read(8_000_001)
        items = parse_tei(content)
        result.update(status="ready" if items else "empty", items=items, url=url)
    except Exception as exc:
        # Exceptions from network clients may include credentialed URLs.
        result["error"] = type(exc).__name__
    workspace.write_json("research/openalex_fulltext.json", result)
    return result


def search_scholarly_snippets(question: str, workspace) -> list[dict]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", os.getenv("S2_API_KEY", "")).strip()
    if not key:
        return []
    import hashlib

    digest = hashlib.sha256(question.encode()).hexdigest()[:20]
    path = workspace.path(f"research/snippets/{digest}.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        query = urllib.parse.urlencode({"query": question[:500], "limit": 5})
        data = scholarly_json_get(
            "https://api.semanticscholar.org/graph/v1/snippet/search?" + query, {"x-api-key": key}
        )
        workspace.write_json(f"research/snippets/{digest}.json", data)
    items = data.get("data", [])
    return [
        dict(item, id=f"S2SNIPPET:{i + 1}")
        for i, item in enumerate(items[:5])
        if isinstance(item, dict)
    ]

"""Opt-in, version-pinned structured arXiv evidence; the uploaded PDF stays primary."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from html.parser import HTMLParser
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from paper_agent.workspace import Workspace

ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})v[1-9]\d*"
STAMP = re.compile(r"arXiv:\s*(" + ARXIV_ID + r")\b", re.I)
MAX_BYTES = 8_000_000
CACHE = "parsed/arxiv_source.json"
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def document_key(parsed) -> str:
    return hashlib.sha256(parsed.full_text.encode("utf-8")).hexdigest()


def paper_version(parsed) -> str | None:
    # Never infer the version from a filename, bibliography, title search or latest redirect.
    matches = set(STAMP.findall(parsed.page_text.get(1, "")))
    return next(iter(matches)) if len(matches) == 1 else None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("arXiv redirected the version-pinned request; source not used.")


def fetch_html(url: str) -> str:
    global _LAST_REQUEST
    with _LOCK:
        time.sleep(max(0.0, 3.0 - (time.monotonic() - _LAST_REQUEST)))
        _LAST_REQUEST = time.monotonic()
        request = Request(
            url, headers={"User-Agent": "DeepPaperReader/0.1 (structured-paper-reader)"}
        )
        with build_opener(_NoRedirect()).open(request, timeout=15) as response:
            if response.geturl() != url or "text/html" not in response.headers.get(
                "Content-Type", ""
            ):
                raise ValueError("Unexpected arXiv response URL or content type.")
            content = response.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("arXiv HTML exceeded the ingestion size limit.")
    return content.decode("utf-8", errors="replace")


class ArticleParser(HTMLParser):
    """Read LaTeXML paragraphs/display MathML alttext, never execute HTML or TeX."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.items: list[dict] = []
        self.section = ""
        self.header_text: list[str] = []
        self.pdf_versions: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a":
            match = re.fullmatch(
                r"(?:https://arxiv.org)?/pdf/(" + ARXIV_ID + r")(?:\.pdf)?", attrs.get("href", "")
            )
            if match:
                self.pdf_versions.add(match[1])
        if tag in {"meta", "link", "img", "br", "hr", "input", "source", "wbr"}:
            return
        parent_skip = bool(self.stack and self.stack[-1]["skip"])
        skip = (
            parent_skip
            or tag in {"script", "style", "nav", "footer"}
            or any(c in classes for c in ("ltx_bibliography", "ltx_appendix"))
        )
        article = tag == "article" or bool(self.stack and self.stack[-1]["article"])
        anchor = attrs.get("id") or (self.stack[-1]["anchor"] if self.stack else "")
        kind = "heading" if re.fullmatch(r"h[1-6]", tag) else "paragraph" if tag == "p" else ""
        self.stack.append(
            dict(tag=tag, skip=skip, article=article, anchor=anchor, kind=kind, text=[])
        )
        if tag == "math" and article and not skip:
            latex = attrs.get("alttext", "").strip()
            if latex:
                # Inline notation is kept in paragraph text; only display equations are separate artifacts.
                for node in self.stack[:-1]:
                    if node["kind"]:
                        node["text"].append(f" ${latex}$ ")
                if attrs.get("display") == "block" or any(
                    "ltx_equation" in str(n.get("classes", "")) for n in self.stack
                ):
                    self.items.append(
                        {
                            "kind": "equation",
                            "latex": latex[:6000],
                            "text": latex[:6000],
                            "anchor": anchor,
                            "section": self.section,
                        }
                    )
            self.stack[-1]["skip"] = True
        self.stack[-1]["classes"] = classes

    def handle_endtag(self, tag):
        index = next(
            (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i]["tag"] == tag), None
        )
        if index is None:
            return
        nodes = self.stack[index:]
        del self.stack[index:]
        for node in reversed(nodes):
            value = " ".join("".join(node["text"]).split())
            if node["skip"] or not node["article"] or not value:
                continue
            if node["kind"] == "heading":
                self.section = value
            elif node["kind"] == "paragraph" and len(value) > 30:
                self.items.append(
                    {
                        "kind": "paragraph",
                        "text": value[:6000],
                        "anchor": node["anchor"],
                        "section": self.section,
                    }
                )

    def handle_data(self, data):
        if self.stack and self.stack[-1]["skip"]:
            return
        if not any(node["article"] for node in self.stack):
            self.header_text.append(data)
        for node in self.stack:
            if node["kind"]:
                node["text"].append(data)


def parse_html(html: str, version: str) -> list[dict]:
    if len(html.encode("utf-8")) > MAX_BYTES:
        raise ValueError("arXiv HTML exceeded the ingestion size limit.")
    parser = ArticleParser()
    parser.feed(html)
    stamps = set(STAMP.findall(" ".join(parser.header_text)))
    if stamps != {version} or (parser.pdf_versions and parser.pdf_versions != {version}):
        raise ValueError("HTML identity/version did not match the uploaded PDF stamp.")
    if not parser.items:
        raise ValueError("No structured article content was found.")
    return parser.items[:3000]


def load_source(parsed, workspace: Workspace) -> dict:
    try:
        result = json.loads(workspace.read_text(CACHE))
        if result.get("document_key") == document_key(parsed) and result.get(
            "version"
        ) == paper_version(parsed):
            return result
    except (OSError, ValueError, AttributeError):
        pass
    return {
        "status": "not_loaded" if paper_version(parsed) else "version_unresolved",
        "version": paper_version(parsed),
        "items": [],
    }


def ingest_source(parsed, workspace: Workspace) -> dict:
    cached = load_source(parsed, workspace)
    if cached["status"] == "ready" or time.time() - cached.get("checked_at", 0) < 300:
        return cached
    version = paper_version(parsed)
    result = {
        "status": "version_unresolved",
        "version": version,
        "document_key": document_key(parsed),
        "checked_at": time.time(),
        "items": [],
    }
    if version:
        url = f"https://arxiv.org/html/{version}"
        result["url"] = url
        try:
            items = parse_html(fetch_html(url), version)
            for index, item in enumerate(items):
                item["id"] = f"ARXIV:{index + 1}"
                item["url"] = url + ("#" + quote(item["anchor"], safe="") if item["anchor"] else "")
            result.update(status="ready", items=items)
        except Exception as exc:
            result.update(status="unavailable", error=str(exc)[:400])
    workspace.write_json(CACHE, result)
    return result


def retrieve_source(parsed, workspace: Workspace, query: str, limit: int = 3) -> list[dict]:
    source = load_source(parsed, workspace)
    if source["status"] != "ready":
        return []
    tokens = set(re.findall(r"[a-zA-Z]{3,}", query.lower())) - {
        "the",
        "this",
        "that",
        "paper",
        "what",
        "explain",
    }
    ranked = sorted(
        source["items"],
        key=lambda item: len(
            tokens
            & set(re.findall(r"[a-zA-Z]{3,}", (item["text"] + " " + item["section"]).lower()))
        ),
        reverse=True,
    )
    return [
        dict(item, version=source["version"])
        for item in ranked
        if tokens & set(re.findall(r"[a-zA-Z]{3,}", (item["text"] + " " + item["section"]).lower()))
    ][:limit]

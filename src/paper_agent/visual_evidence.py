"""Bounded visual evidence for questions about plots, tables and diagrams."""

from __future__ import annotations

import base64
import hashlib
import json
import os

from paper_agent.evidence_index import render_source_crop


def visual_messages(parsed, evidence, prompt: str) -> tuple[object, list[str]]:
    if os.getenv("PAPER_CHAT_VISUAL_ENABLED", "true").lower() in {"false", "0", "off"}:
        return prompt, []
    pages = list(
        dict.fromkeys(
            item.page for item in evidence if item.kind in {"figure", "table"} and item.page
        )
    )[:2]
    content = [
        {
            "type": "text",
            "text": prompt
            + "\nAttached PDF pages are primary visual evidence. Inspect axes, units, legends, headers, "
            "footnotes and conditions. Cite the corresponding FIG/TABLE ID. Never invent unreadable values. "
            "Derived comparisons must be labeled inference and show their calculation.",
        }
    ]
    attached = []
    for page in pages:
        try:
            png = render_source_crop(parsed, page, scale=1.4)
            content.append({"type": "text", "text": f"Original uploaded PDF page {page}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                }
            )
            attached.append(str(page))
        except (OSError, ValueError, RuntimeError):
            continue
    return ([{"role": "user", "content": content}] if attached else prompt), attached


def visual_cache_key(parsed, question, evidence, model):
    payload = [
        "visual-v1",
        hashlib.sha256(parsed.full_text.encode()).hexdigest(),
        question,
        model,
        [(e.id, e.page, e.text) for e in evidence],
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

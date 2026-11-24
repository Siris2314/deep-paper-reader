"""Headless integration smoke test against an explicitly started local reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8517")
    parser.add_argument("--output", type=Path, default=Path("tmp/reader-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("Boolean(state.workspace)", timeout=30000)
        health = page.request.get(args.url + "/api/health")
        assert health.ok
        assert health.json()["build"].endswith("v37")
        page.evaluate("""() => {
          const panel = $('evidencePanel');
          panel.dataset.mathKey = 'smoke';
          panel.dataset.cropUrl = `/api/source-crop?workspace=${encodeURIComponent(state.workspace)}&page=1`;
          const example = {equation_id:'smoke', page:1, status:'transcription_review_required',
            display_latex:'x = y', transcription_confidence:0.4,
            transcription_candidates:{pdf_geometry:'x = y', vision_math_agent:'x = z'},
            transcription_warnings:['Candidates disagree.'], parse:{}, evaluation:{}};
          state.mathExplanations.set('smoke', example);
          setSidePanel('evidence'); renderMathPopover(example, 'smoke');
        }""")
        assert page.get_by_text("Explanation withheld.", exact=False).is_visible()
        page.get_by_text("Correct the equation notation", exact=True).click()
        assert page.get_by_role("textbox", name="Corrected equation LaTeX").is_visible()
        page.screenshot(path=str(args.output / "equation-review.png"), full_page=True)
        page.evaluate("""() => {
          setSidePanel('chat'); appendChatThinking('Checking claims', {
            answer:'A draft answer with source checks still pending.',
            citations:[{id:'P1.1',page:1,label:'Page 1',source:'paper',excerpt:'Supporting source text',bbox:[30,30,300,100],page_width:612,page_height:792}]
          });
        }""")
        assert page.get_by_text("Draft — source checks pending").is_visible()
        page.screenshot(path=str(args.output / "answer-preview.png"), full_page=True)
        page.locator('[data-evidence-key="preview:P1.1"]').click()
        assert page.locator(".source-focus").count() == 1
        page.screenshot(path=str(args.output / "citation-focus.png"), full_page=True)
        crop = page.request.get(
            args.url + "/api/source-crop",
            params={
                "workspace": page.evaluate("state.workspace"),
                "page": 1,
                "bbox": json.dumps([30, 30, 300, 100]),
            },
        )
        assert crop.ok and crop.body().startswith(b"\x89PNG")
        assert not errors, errors
        browser.close()
    print(
        "UI smoke passed: equation review, correction form, draft preview, health and actual PDF crop."
    )


if __name__ == "__main__":
    main()

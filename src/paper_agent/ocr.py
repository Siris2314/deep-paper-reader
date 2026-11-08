from __future__ import annotations

import os
from typing import Any

import fitz


def _ocr_dpi() -> int:
    try:
        return max(150, min(300, int(os.getenv("PAPER_OCR_DPI", "200"))))
    except ValueError:
        return 200


def _image_coverage(page: fitz.Page) -> float:
    area = max(page.rect.width * page.rect.height, 1.0)
    return min(
        1.0,
        sum((fitz.Rect(item["bbox"]) & page.rect).get_area() for item in page.get_image_info())
        / area,
    )


def should_ocr_page(page: fitz.Page, native_text: str) -> bool:
    if os.getenv("PAPER_OCR_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
        return False
    readable = sum(char.isalnum() for char in native_text)
    corrupt = native_text.count("\ufffd")
    if corrupt >= 3 and corrupt * 5 >= len(native_text.strip()):
        return True
    if readable >= 80:
        return False
    if _image_coverage(page) >= 0.35:
        return True
    return readable == 0 and len(page.get_drawings()) >= 20


def _make_textpage(page: fitz.Page) -> fitz.TextPage:
    return page.get_textpage_ocr(
        language=os.getenv("PAPER_OCR_LANGUAGE", "eng"), dpi=_ocr_dpi(), full=True
    )


def extract_page_text(page: fitz.Page) -> tuple[str, dict[str, Any] | None, str | None]:
    native_text = page.get_text("text") or ""
    if not should_ocr_page(page, native_text):
        return native_text.strip(), None, None

    try:
        textpage = _make_textpage(page)
        ocr_text = page.get_text("text", textpage=textpage, sort=True).strip()
        if not ocr_text:
            return native_text.strip(), None, "OCR found no text"
        layout = {
            "words": page.get_text("words", textpage=textpage, sort=True),
            "dict": page.get_text("dict", textpage=textpage, sort=True),
        }
        return ocr_text, layout, "ocr"
    except (RuntimeError, ValueError) as exc:
        return native_text.strip(), None, f"OCR unavailable: {exc}"

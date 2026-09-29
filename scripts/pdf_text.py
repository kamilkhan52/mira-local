"""PDF text extraction with permissive licenses only.

pypdfium2 (Apache-2.0 / BSD-3, bundles Google's PDFium) is the primary engine;
pypdf (BSD-3) is the fallback. PyMuPDF is deliberately NOT used: it is AGPL-3.0
(or a paid Artifex license), which is not acceptable for internal enterprise use.
"""
from __future__ import annotations

from io import BytesIO

try:
    import pypdfium2 as pdfium  # type: ignore
except ImportError:  # pragma: no cover - exercised only without pypdfium2
    pdfium = None

from pypdf import PdfReader


def page_texts(src, max_pages: int | None = None) -> tuple[list[str], int]:
    """(text of the first max_pages pages, total page count). src is a path,
    bytes, or a BytesIO."""
    if isinstance(src, BytesIO):
        src = src.getvalue()
    if pdfium is not None:
        try:
            pdf = pdfium.PdfDocument(src)
            try:
                n = len(pdf)
                texts = []
                for i in range(n if max_pages is None else min(n, max_pages)):
                    page = pdf[i]
                    tp = page.get_textpage()
                    texts.append(tp.get_text_range())
                    tp.close()
                    page.close()
                return texts, n
            finally:
                pdf.close()
        except Exception:  # noqa: BLE001 — fall back to pypdf on any PDFium failure
            pass
    reader = PdfReader(BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    return [(p.extract_text() or "") for p in pages], len(reader.pages)

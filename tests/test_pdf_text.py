"""pypdfium2 must be safe under the extractors' thread pools (PDFium itself
segfaults on concurrent use)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pdf_text  # noqa: E402


def _pdf_bytes(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    import io
    w = PdfWriter()
    page = w.add_blank_page(width=300, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
        {NameObject("/F1"): w._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_extracts_first_page_text():
    texts, n = pdf_text.page_texts(_pdf_bytes("Samsung Electronics"), max_pages=1)
    assert n == 1 and "Samsung Electronics" in texts[0]


def test_concurrent_extraction_does_not_crash():
    data = _pdf_bytes("Micron Technology")
    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: pdf_text.page_texts(data, max_pages=1), range(400)))
    assert all("Micron Technology" in t[0][0] for t in results)

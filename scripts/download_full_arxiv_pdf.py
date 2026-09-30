import sys
import os
import threading
# Scratch dir for PDFs/results: MIRA_TEMP_DIR, else <MIRA_DATA_DIR or repo/data>/temp.
_TEMP = os.environ.get("MIRA_TEMP_DIR") or os.path.join(
    os.environ.get("MIRA_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), "temp")
import requests
import io
import json
from datetime import date
from contextlib import contextmanager

from pdf_text import page_texts  # pypdfium2/pypdf only; PyMuPDF (AGPL) is not used

# Cache directory - same as first-page extraction
CACHE_DIR = os.path.join(_TEMP, "full_pdfs")
PDF_LIBRARY_DIR = os.path.join(_TEMP, "pdf_library")
FULL_TEXT_CACHE_DIR = os.path.join(_TEMP, "full_text_cache")

_STDERR_LOCK = threading.Lock()


@contextmanager
def suppress_c_stderr():
    """
    Context manager to suppress C-level stderr (fd 2) output.
    This silences warnings from libraries like MuPDF/Ghostscript.
    """
    # Serialized: fd 2 is process-wide; concurrent save/redirect/restore can
    # leave stderr pointing at /dev/null for good.
    with _STDERR_LOCK, open(os.devnull, 'w') as devnull:
        original_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stderr_fd)

def sanitize_text(text: str) -> str:
    """Remove control characters and problematic Unicode that can break JSON/LLM processing."""
    if not text:
        return ""
    cleaned = []
    for char in text:
        code = ord(char)
        if code > 0xFFFF:  # Beyond BMP (mathematical symbols, emoji, etc.)
            continue  # Skip these entirely
        elif code >= 32 or char in '\n\r\t':
            cleaned.append(char)
        elif code == 0x0C:  # Form feed -> newline
            cleaned.append('\n')
    return ''.join(cleaned)

def ensure_cache_dirs():
    os.makedirs(PDF_LIBRARY_DIR, exist_ok=True)
    os.makedirs(FULL_TEXT_CACHE_DIR, exist_ok=True)

def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ID/URL to a canonical ID (keeps version if provided)."""
    if 'arxiv.org/abs/' in arxiv_id:
        clean = arxiv_id.split('arxiv.org/abs/')[-1]
    elif 'arxiv.org/pdf/' in arxiv_id:
        clean = arxiv_id.split('arxiv.org/pdf/')[-1]
    else:
        clean = arxiv_id
    return clean.replace('.pdf', '')

def make_safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace('/', '_')

def get_pdf_library_path(safe_id: str) -> str:
    return os.path.join(PDF_LIBRARY_DIR, f"{safe_id}.pdf")

def get_full_text_cache_path(safe_id: str) -> str:
    return os.path.join(FULL_TEXT_CACHE_DIR, f"{safe_id}.json")

def atomic_write_json(path: str, data: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

def download_pdf_to_library(pdf_url: str, pdf_path: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
    response.raise_for_status()
    tmp_path = f"{pdf_path}.download"
    with open(tmp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, pdf_path)

def check_cache(arxiv_id: str) -> dict | None:
    """Check if cached result exists for this paper."""
    paper_id = normalize_arxiv_id(arxiv_id)
    safe_id = make_safe_id(paper_id)
    cache_path = get_full_text_cache_path(safe_id)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                result = json.load(f)
                result['cached'] = True
                return result
        except Exception:
            pass
    return None

def save_cache(arxiv_id: str, result: dict):
    """Save result to cache."""
    paper_id = normalize_arxiv_id(arxiv_id)
    safe_id = make_safe_id(paper_id)
    cache_path = get_full_text_cache_path(safe_id)
    try:
        atomic_write_json(cache_path, result)
    except Exception:
        pass  # Silently fail cache writes

def download_and_extract(arxiv_id, use_cache=True):
    ensure_cache_dirs()
    # Check cache first (unless disabled)
    if use_cache:
        cached = check_cache(arxiv_id)
        if cached:
            return cached
    
    try:
        # Extract arXiv ID
        paper_id = normalize_arxiv_id(arxiv_id)

        # Download PDF
        pdf_url = f"http://export.arxiv.org/pdf/{paper_id}.pdf"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        safe_id = make_safe_id(paper_id)
        pdf_path = get_pdf_library_path(safe_id)

        # Ensure PDF exists in shared library
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0 or not use_cache:
            response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "full_text": "",
                    "page_count": 0,
                    "pdf_url": pdf_url,
                    "cached": False
                }
            tmp_path = f"{pdf_path}.download"
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, pdf_path)

        # Extract text from all pages
        full_text = ""
        with suppress_c_stderr():
            texts, page_count = page_texts(pdf_path)
        for text in texts:
            full_text += text + "\n\n"
        
        # Sanitize the extracted text
        full_text = sanitize_text(full_text)
        
        result = {
            "success": True,
            "full_text": full_text,
            "page_count": page_count,
            "pdf_url": pdf_url,
            "error": "",
            "cached": False
        }
        
        # Save to cache
        if use_cache:
            save_cache(arxiv_id, result)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "full_text": "",
            "page_count": 0,
            "pdf_url": "",
            "cached": False
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download and extract full text from ArXiv PDF')
    parser.add_argument('arxiv_id', help='ArXiv ID or URL')
    parser.add_argument('--no-cache', action='store_true', help='Ignore cached results and force fresh download')
    args = parser.parse_args()
    
    ensure_cache_dirs()

    if args.no_cache:
        print(f"🔄 Cache disabled, forcing fresh download", file=sys.stderr)
    
    result = download_and_extract(args.arxiv_id, use_cache=not args.no_cache)
    print(json.dumps(result))
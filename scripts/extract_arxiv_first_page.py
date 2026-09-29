#!/usr/bin/env python3
import sys
import json
import asyncio
import time
import os
# Scratch dir for PDFs/results: MIRA_TEMP_DIR, else <MIRA_DATA_DIR or repo/data>/temp.
_TEMP = os.environ.get("MIRA_TEMP_DIR") or os.path.join(
    os.environ.get("MIRA_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), "temp")
import hashlib
from datetime import date
from io import BytesIO
from typing import List, Dict, Any
from contextlib import contextmanager
from contextlib import contextmanager

try:
    import requests
except ImportError as e:
    print(json.dumps({"success": False, "error": f"Missing dependency: {e}"}))
    sys.exit(1)

from pdf_text import page_texts  # pypdfium2/pypdf only; PyMuPDF (AGPL) is not used

# Configuration
DEFAULT_CONCURRENCY = 3  # Conservative default
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds
TIMEOUT = 30  # seconds per request
OUTPUT_DIR = _TEMP
PDF_LIBRARY_DIR = os.path.join(_TEMP, "pdf_library")
FIRST_PAGE_CACHE_DIR = os.path.join(_TEMP, "first_page_cache")

@contextmanager
def suppress_c_stderr():
    """
    Context manager to suppress C-level stderr (fd 2) output.
    This silences warnings from libraries like MuPDF/Ghostscript.
    """
    with open(os.devnull, 'w') as devnull:
        original_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stderr_fd)

@contextmanager
def suppress_c_stderr():
    """
    Context manager to suppress C-level stderr (fd 2) output.
    This silences warnings from libraries like MuPDF/Ghostscript.
    """
    with open(os.devnull, 'w') as devnull:
        original_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stderr_fd)

def get_cache_filename(count: int) -> str:
    """Generate a cache filename based on today's date and paper count"""
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(OUTPUT_DIR, f"pdf_cache_{date_str}_{count}.json")

def check_cache(count: int) -> str | None:
    """Check if cached results exist for today with this paper count. Returns filepath if exists."""
    cache_file = get_cache_filename(count)
    if os.path.exists(cache_file):
        return cache_file
    return None

def sanitize_text(text: str) -> str:
    """
    Remove control characters and problematic Unicode that can break JSON/LLM processing.
    Keeps: printable ASCII, newlines, tabs, and common Unicode (BMP only).
    """
    if not text:
        return ""
    # Remove:
    # - Control characters (0x00-0x1F) except \n, \r, \t
    # - DEL (0x7F)
    # - Characters beyond BMP (> 0xFFFF) - mathematical symbols that cause issues
    cleaned = []
    for char in text:
        code = ord(char)
        if code > 0xFFFF:  # Beyond BMP (mathematical symbols, emoji, etc.)
            continue  # Skip these entirely
        elif code >= 32 or char in '\n\r\t':  # Printable or allowed whitespace
            cleaned.append(char)
        elif code == 0x0C:  # Form feed - replace with newline
            cleaned.append('\n')
        # else: skip control characters (null bytes, bell, etc.)
    return ''.join(cleaned)

def log_progress(message: str):
    """Log to stderr so it doesn't interfere with JSON output"""
    print(message, file=sys.stderr, flush=True)

def ensure_output_dir():
    """Ensure the output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_cache_dirs():
    """Ensure shared PDF library and cache directories exist"""
    os.makedirs(PDF_LIBRARY_DIR, exist_ok=True)
    os.makedirs(FIRST_PAGE_CACHE_DIR, exist_ok=True)

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
    """Filename-safe ID for cache keys and library storage."""
    return arxiv_id.replace('/', '_')

def get_pdf_library_path(safe_id: str) -> str:
    return os.path.join(PDF_LIBRARY_DIR, f"{safe_id}.pdf")

def get_first_page_cache_path(safe_id: str) -> str:
    return os.path.join(FIRST_PAGE_CACHE_DIR, f"{safe_id}.json")

def atomic_write_json(path: str, data: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)

def download_pdf_to_library(pdf_url: str, pdf_path: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(pdf_url, headers=headers, timeout=TIMEOUT, stream=True)
    response.raise_for_status()
    tmp_path = f"{pdf_path}.download"
    with open(tmp_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, pdf_path)

def get_batch_signature(arxiv_ids: List[str]) -> str:
    """Create a short stable signature for a specific batch content/order."""
    payload = "\n".join(str(x) for x in arxiv_ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def get_cache_filename(arxiv_ids: List[str]) -> str:
    """Generate a deterministic cache filename based on date, count, and batch signature."""
    today = date.today().isoformat()  # e.g., "2025-12-04"
    count = len(arxiv_ids)
    signature = get_batch_signature(arxiv_ids)
    return os.path.join(OUTPUT_DIR, f"pdf_results_{today}_{count}_{signature}.json")

def check_cache(arxiv_ids: List[str]) -> str | None:
    """Check if cached results exist for today's exact batch. Returns path if exists."""
    cache_path = get_cache_filename(arxiv_ids)
    if os.path.exists(cache_path):
        return cache_path
    return None

def extract_pdf_metadata(arxiv_id: str, pdf_url: str = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract metadata from a single ArXiv PDF.
    This is the original synchronous function, kept for backward compatibility.
    """
    # Construct PDF URL
    if pdf_url is None:
        paper_id = normalize_arxiv_id(arxiv_id)
        pdf_url = f"http://export.arxiv.org/pdf/{paper_id}.pdf"
    else:
        paper_id = normalize_arxiv_id(arxiv_id)

    safe_id = make_safe_id(paper_id)
    cache_path = get_first_page_cache_path(safe_id)

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            cached['id'] = arxiv_id
            cached['pdf_url'] = pdf_url
            cached['cached'] = True
            return cached
        except Exception:
            pass

    pdf_path = get_pdf_library_path(safe_id)

    try:
        # Ensure PDF exists in shared library
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0 or not use_cache:
            download_pdf_to_library(pdf_url, pdf_path)

        # Open PDF
        with suppress_c_stderr():
            texts, page_count = page_texts(pdf_path, max_pages=1)
        first_page_text = sanitize_text(texts[0] if texts else "")

        result = {
            "id": arxiv_id,
            "success": True,
            "first_page_text": first_page_text,
            "page_count": page_count,
            "pdf_url": pdf_url,
            "cached": False
        }
        if use_cache:
            try:
                atomic_write_json(cache_path, result)
            except Exception:
                pass
        return result
    except requests.exceptions.HTTPError as e:
        return {
            "id": arxiv_id,
            "success": False,
            "first_page_text": "",
            "page_count": 0,
            "pdf_url": pdf_url,
            "error": f"HTTP {e.response.status_code}: {str(e)}",
            "status_code": e.response.status_code
        }
    except Exception as e:
        return {
            "id": arxiv_id,
            "success": False,
            "first_page_text": "",
            "page_count": 0,
            "pdf_url": pdf_url,
            "error": str(e)
        }

async def download_with_retry(arxiv_id: str, semaphore: asyncio.Semaphore,
                               rate_limit_event: asyncio.Event, use_cache: bool = True) -> Dict[str, Any]:
    """
    Download a single PDF with retry logic and rate limit handling.
    Uses semaphore to control concurrency.
    """
    # Construct PDF URL
    if 'arxiv.org/abs/' in arxiv_id:
        paper_id = arxiv_id.split('arxiv.org/abs/')[-1]
    else:
        paper_id = arxiv_id

    pdf_url = f"http://export.arxiv.org/pdf/{paper_id}.pdf"

    backoff = INITIAL_BACKOFF
    last_error = None

    for attempt in range(MAX_RETRIES):
        # Wait if we've detected rate limiting
        if rate_limit_event.is_set():
            wait_time = min(backoff * (2 ** attempt), MAX_BACKOFF)
            log_progress(f"⏳ Rate limit detected, waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait_time)

        async with semaphore:  # Control concurrency
            try:
                # Run the synchronous download in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    extract_pdf_metadata,
                    arxiv_id,
                    pdf_url,
                    use_cache
                )

                # Check for rate limiting
                if not result["success"] and result.get("status_code") in [429, 503]:
                    log_progress(f"⚠️  Rate limit detected for {arxiv_id} (HTTP {result['status_code']})")
                    rate_limit_event.set()  # Signal other tasks
                    last_error = result
                    continue  # Retry

                # Retry transient non-HTTP failures (e.g. IncompleteRead, connection reset)
                if not result["success"]:
                    last_error = result
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF)
                        continue
                    return result

                # Clear rate limit flag on success
                if result["success"] and rate_limit_event.is_set():
                    rate_limit_event.clear()
                    log_progress("✅ Rate limit cleared, resuming normal speed")

                return result

            except Exception as e:
                last_error = {
                    "id": arxiv_id,
                    "success": False,
                    "first_page_text": "",
                    "page_count": 0,
                    "pdf_url": pdf_url,
                    "error": f"Attempt {attempt + 1} failed: {str(e)}"
                }

                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)

    # All retries failed
    return last_error if last_error else {
        "id": arxiv_id,
        "success": False,
        "first_page_text": "",
        "page_count": 0,
        "pdf_url": pdf_url,
        "error": "All retry attempts failed"
    }

async def process_batch(arxiv_ids: List[str], concurrency: int = DEFAULT_CONCURRENCY, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Process multiple ArXiv IDs concurrently with controlled concurrency.
    """
    log_progress(f"🚀 Starting batch processing: {len(arxiv_ids)} papers with concurrency={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_event = asyncio.Event()  # Shared flag for rate limiting

    start_time = time.time()

    # Create tasks for all downloads
    tasks = [
        download_with_retry(arxiv_id, semaphore, rate_limit_event, use_cache)
        for arxiv_id in arxiv_ids
    ]

    # Process with progress updates
    results = []
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        results.append(result)

        status = "✅" if result["success"] else "❌"
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(arxiv_ids) - i) / rate if rate > 0 else 0

        log_progress(f"{status} [{i}/{len(arxiv_ids)}] {result['id']} | "
                    f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["success"])

    log_progress(f"\n✨ Batch complete: {success_count}/{len(arxiv_ids)} successful in {total_time:.1f}s "
                f"(avg {total_time/len(arxiv_ids):.1f}s per paper)")

    return results

def main():
    """Main entry point supporting both single and batch modes"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract metadata from ArXiv PDFs')
    parser.add_argument('arxiv_id', nargs='?', help='Single ArXiv ID (for backward compatibility)')
    parser.add_argument('--batch', action='store_true', help='Batch mode: read JSON array from stdin')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                       help=f'Max concurrent downloads in batch mode (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--output-file', action='store_true', 
                       help='Write results to temp file instead of stdout (for large outputs)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Ignore cached results and force fresh extraction')

    args = parser.parse_args()

    ensure_output_dir()
    ensure_cache_dirs()

    # Batch mode
    if args.batch:
        try:
            # Read JSON array from stdin
            input_data = json.loads(sys.stdin.read())

            if not isinstance(input_data, list):
                print(json.dumps({"success": False, "error": "Batch mode expects JSON array"}))
                sys.exit(1)

            # Check for cached results (same date + same count) unless --no-cache
            if args.output_file:
                if not args.no_cache:
                    cached = check_cache(input_data)
                    if cached:
                        log_progress(f"📦 Cache hit! Using existing results: {cached}")
                        print(json.dumps({"output_file": cached, "cached": True}))
                        sys.exit(0)
            
            if args.no_cache:
                log_progress("🔄 Cache disabled, forcing fresh extraction")

            # Run async batch processing
            results = asyncio.run(process_batch(input_data, args.concurrency, use_cache=not args.no_cache))
            
            # Write to file if --output-file flag is set
            if args.output_file:
                output_path = get_cache_filename(input_data)
                with open(output_path, 'w') as f:
                    json.dump(results, f)
                log_progress(f"💾 Results cached to: {output_path}")
                # Only output the file path
                print(json.dumps({"output_file": output_path, "cached": False}))
            else:
                print(json.dumps(results))

        except json.JSONDecodeError as e:
            print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
            sys.exit(1)
        except Exception as e:
            print(json.dumps({"success": False, "error": f"Batch processing failed: {e}"}))
            sys.exit(1)

    # Single mode (backward compatible)
    else:
        if not args.arxiv_id:
            print(json.dumps({"success": False, "error": "No arxiv_id provided"}))
            sys.exit(1)

        result = extract_pdf_metadata(args.arxiv_id, use_cache=not args.no_cache)
        print(json.dumps(result))

if __name__ == "__main__":
    main()
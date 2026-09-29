#!/usr/bin/env python3
import sys
import json
import asyncio
import time
import os
# Scratch dir for PDFs/results: MIRA_TEMP_DIR, else <MIRA_DATA_DIR or repo/data>/temp.
_TEMP = os.environ.get("MIRA_TEMP_DIR") or os.path.join(
    os.environ.get("MIRA_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), "temp")
from datetime import date
from io import BytesIO
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import requests

from pdf_text import page_texts  # pypdfium2/pypdf only; PyMuPDF (AGPL) is not used

# Configuration
DEFAULT_CONCURRENCY = 3  # Conservative default
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds
TIMEOUT = 30  # seconds per request
OUTPUT_DIR = _TEMP

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
    Remove control characters that can break JSON parsing and LLM processing.
    Keeps: printable ASCII, newlines, tabs, and common Unicode.
    """
    if not text:
        return ""
    # Remove all control characters (0x00-0x1F) except \n (0x0A), \r (0x0D), \t (0x09)
    # Also remove 0x7F (DEL) and other problematic chars
    cleaned = []
    for char in text:
        code = ord(char)
        if code >= 32 or char in '\n\r\t':  # Printable or allowed whitespace
            cleaned.append(char)
        elif code == 0x0C:  # Form feed - replace with newline
            cleaned.append('\n')
        # else: skip the character (null bytes, bell, etc.)
    return ''.join(cleaned)

def log_progress(message: str):
    """Log to stderr so it doesn't interfere with JSON output"""
    print(message, file=sys.stderr, flush=True)

def ensure_output_dir():
    """Ensure the output directory exists"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_cache_filename(count: int) -> str:
    """Generate a deterministic cache filename based on date and paper count"""
    today = date.today().isoformat()  # e.g., "2025-12-04"
    return os.path.join(OUTPUT_DIR, f"pdf_results_{today}_{count}.json")

def check_cache(count: int) -> str | None:
    """Check if cached results exist for today's date and paper count. Returns path if exists."""
    cache_path = get_cache_filename(count)
    if os.path.exists(cache_path):
        return cache_path
    return None

def extract_pdf_metadata(arxiv_id: str, pdf_url: str = None) -> Dict[str, Any]:
    """
    Extract metadata from a single ArXiv PDF.
    This is the original synchronous function, kept for backward compatibility.
    """
    # Construct PDF URL
    if pdf_url is None:
        if 'arxiv.org/abs/' in arxiv_id:
            paper_id = arxiv_id.split('arxiv.org/abs/')[-1]
        else:
            paper_id = arxiv_id
        pdf_url = f"http://export.arxiv.org/pdf/{paper_id}.pdf"

    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=TIMEOUT)
        response.raise_for_status()

        # Open PDF
        pdf_bytes = BytesIO(response.content)

        texts, page_count = page_texts(pdf_bytes, max_pages=1)
        first_page_text = sanitize_text(texts[0] if texts else "")

        return {
            "id": arxiv_id,
            "success": True,
            "first_page_text": first_page_text,
            "page_count": page_count,
            "pdf_url": pdf_url
        }
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
                               rate_limit_event: asyncio.Event,
                               executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, Any]:
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
                    executor,
                    extract_pdf_metadata,
                    arxiv_id,
                    pdf_url
                )

                # Check for rate limiting
                if not result["success"] and result.get("status_code") in [429, 503]:
                    log_progress(f"⚠️  Rate limit detected for {arxiv_id} (HTTP {result['status_code']})")
                    rate_limit_event.set()  # Signal other tasks
                    last_error = result
                    continue  # Retry

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

async def process_batch(arxiv_ids: List[str], concurrency: int = DEFAULT_CONCURRENCY,
                        executor: Optional[ThreadPoolExecutor] = None) -> List[Dict[str, Any]]:
    """
    Process multiple ArXiv IDs concurrently with controlled concurrency.
    """
    log_progress(f"🚀 Starting batch processing: {len(arxiv_ids)} papers with concurrency={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_event = asyncio.Event()  # Shared flag for rate limiting

    start_time = time.time()

    # Create tasks for all downloads
    tasks = [
        download_with_retry(arxiv_id, semaphore, rate_limit_event, executor)
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

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        try:
            # Read JSON array from stdin
            input_data = json.loads(sys.stdin.read())

            if not isinstance(input_data, list):
                print(json.dumps({"success": False, "error": "Batch mode expects JSON array"}))
                sys.exit(1)

            # Check for cached results (same date + same count)
            if args.output_file:
                ensure_output_dir()
                cached = check_cache(len(input_data))
                if cached:
                    log_progress(f"📦 Cache hit! Using existing results: {cached}")
                    print(json.dumps({"output_file": cached, "cached": True}))
                    sys.exit(0)

            # Run async batch processing
            results = asyncio.run(process_batch(input_data, args.concurrency))
            
            # Write to file if --output-file flag is set
            if args.output_file:
                output_path = get_cache_filename(len(input_data))
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

        result = extract_pdf_metadata(args.arxiv_id)
        print(json.dumps(result))

if __name__ == "__main__":
    main()
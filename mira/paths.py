"""Filesystem layout. Code and config live in the repo; everything a run
produces or caches lives under DATA_DIR (default ./data, override with
MIRA_DATA_DIR) so it can sit on a separate volume and never enters git.

  data/report-files/   reports (prod/, tests/), LLM stage cache (cache/<profile>/<stage>/),
                       hypotheses/ — same layout the n8n deployment used, so an
                       existing report-files/ tree can be copied in unchanged
  data/cache/          local JSON caches and venue corpora
  data/temp/           crawler outputs, PDFs, full-text cache
  data/lightrag/       LightRAG working dirs (optional graph)
  data/realtime-state/ realtime monitor ledgers and alerts
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("MIRA_CONFIG_DIR", ROOT / "configs"))
DATA_DIR = Path(os.environ.get("MIRA_DATA_DIR", ROOT / "data"))
REPORT_FILES = DATA_DIR / "report-files"
LOCAL_CACHE = DATA_DIR / "cache"
TEMP_DIR = DATA_DIR / "temp"
LIGHTRAG_DIR = DATA_DIR / "lightrag"
REALTIME_DIR = DATA_DIR / "realtime-state"
SCRIPTS_DIR = ROOT / "scripts"
CRAWLERS_DIR = ROOT / "crawlers"

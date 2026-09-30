#!/usr/bin/env python3
"""Weekly head-to-head: n8n vs mira-local at each Jev level, plus the live agent.

Runs sequentially so each run's cost can be measured from the OpenRouter
balance (ground truth) as well as mira's own per-call ledger:

  1. n8n (webhook, test mode, cache bypass)
  2. mira all-LLM, 3. mira Jev prescreen, 4. mira Jev gate, 5. mira Jev replace
     (Prefect deployment digest-manual — the production path; LLM cache
     bypassed so LLM stages start cold; first-page PDFs warm for every run)
  6-7. live agent (news) twice: first run and the steady state

Needs: n8n container up; `make server` and `make serve` running.
  bin/mira-env python tools/experiment_weekly.py --current-date 2026-09-28
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mira.paths import DATA_DIR, TEMP_DIR  # noqa: E402

OUT = DATA_DIR / "experiment"
N8N_WEBHOOK = "http://localhost:5678/webhook/mira-backfill"


def balance() -> float:
    key = dotenv_values(ROOT / ".env").get("MIRA_LLM_API_KEY")
    d = requests.get("https://openrouter.ai/api/v1/credits",
                     headers={"Authorization": f"Bearer {key}"}, timeout=30).json()["data"]
    return d["total_credits"] - d["total_usage"]


def settled_balance(prev: float | None = None, wait: int = 45, tries: int = 6) -> float:
    """OpenRouter's usage total lags a little; read until two reads agree."""
    last = None
    for _ in range(tries):
        time.sleep(wait)
        b = balance()
        if last is not None and abs(b - last) < 0.005:
            return b
        last = b
    return last


def n8n_latest_exec() -> dict:
    q = ("import sqlite3,json;c=sqlite3.connect('file:/home/node/.n8n/database.sqlite?mode=ro',uri=True);"
         "r=c.execute(\"select id,status,startedAt,stoppedAt from execution_entity where "
         "workflowId='iz3yMcSlkWIQhRmn' order by id desc limit 1\").fetchone();print(json.dumps(r))")
    r = json.loads(subprocess.run(["docker", "exec", "n8n", "python3", "-c", q],
                                  capture_output=True, text=True, check=True).stdout)
    return {"id": r[0], "status": r[1], "startedAt": r[2], "stoppedAt": r[3]}


def clear_pdf_caches() -> None:
    for sub in ("pdf_library", "first_page_cache", "full_text_cache", "full_pdfs"):
        shutil.rmtree(TEMP_DIR / sub, ignore_errors=True)


def n8n_cache(action: str) -> None:
    """Cold n8n run that still writes its cache (so a retry after n8n's
    post-selection crash only repeats the later stages): set the profile's
    cache aside, then merge the new entries back into it afterwards."""
    base = "/report-files/cache/memory-innovation"
    if action == "aside":
        cmd = f"[ -d {base}.pre-exp ] || mv {base} {base}.pre-exp; mkdir -p {base}"
    else:  # restore: keep the original entries, add the new ones
        cmd = (f"[ -d {base}.pre-exp ] && cp -Rn {base}/. {base}.pre-exp/ && rm -rf {base} "
               f"&& mv {base}.pre-exp {base}")
    subprocess.run(["docker", "exec", "n8n", "sh", "-c", cmd], check=True)


def run_n8n(args, retry: bool = False) -> dict:
    payload = {"triggerNode": "Memory Weekly Trigger", "current_date_override": args.current_date,
               "lookback_days_override": 8, "trend_enabled_override": True,
               "test_mode_override": True, "llm_cache_bypass": False, "max_limit": 2000}
    b0 = balance()
    t0 = time.time()
    r = requests.post(N8N_WEBHOOK, json=payload, timeout=6 * 3600)
    wall = time.time() - t0
    ex = n8n_latest_exec()
    b1 = settled_balance(b0)
    subprocess.run([sys.executable, str(ROOT / "tools" / "n8n_export.py"), str(ex["id"])], check=False)
    return {"run": "n8n", "http": r.status_code, "response": r.text[:300], "execution": ex,
            "wall_seconds": round(wall, 1), "cost_measured_usd": round(b0 - b1, 4)}


def run_mira(args, level: str) -> dict:
    from prefect.deployments import run_deployment
    # PDFs stay warm for every run (n8n's container has its own warm cache):
    # equal footing without re-downloading ~2,000 PDFs from arXiv per run.
    # Cold PDF times are reported separately. LLM caches are bypassed.
    tag = {"off": "[mira · all LLM]", "prescreen": "[mira · Jev prescreen]",
           "gate": "[mira · Jev gate]", "replace": "[mira · Jev replace]"}[level]
    params = {"profile": "memory-innovation", "mode": "weekly", "current_date": args.current_date,
              "lookback_days": 8, "max_limit": 2000, "test_mode": True, "llm_cache_bypass": True,
              "trend_enabled": True, "jev_level": level, "subject_tag": tag,
              "recipients": [args.to]}
    b0 = balance()
    t0 = time.time()
    fr = run_deployment("digest/digest-manual", parameters=params, timeout=None)
    wall = time.time() - t0
    b1 = settled_balance(b0)
    result = None
    try:
        result = fr.state.result()
    except Exception as e:  # noqa: BLE001
        result = {"error": repr(e)}
    return {"run": f"mira-{level}", "flow_run": str(fr.id), "state": fr.state.type.value,
            "wall_seconds": round(wall, 1), "cost_measured_usd": round(b0 - b1, 4), "result": result}


def run_live(args, label: str) -> dict:
    from prefect.deployments import run_deployment
    b0 = balance()
    t0 = time.time()
    fr = run_deployment("realtime/realtime-manual", parameters={}, timeout=None)
    wall = time.time() - t0
    b1 = settled_balance(b0, wait=20)
    try:
        result = fr.state.result()
    except Exception as e:  # noqa: BLE001
        result = {"error": repr(e)}
    return {"run": label, "state": fr.state.type.value, "wall_seconds": round(wall, 1),
            "cost_measured_usd": round(b0 - b1, 4), "result": result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-date", default="2026-09-28")
    ap.add_argument("--to", default="kamilkhan52@outlook.com")
    ap.add_argument("--only", nargs="*", help="subset: n8n off prescreen gate replace live")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"weekly_{args.current_date}.json"
    runs = json.loads(path.read_text()) if path.exists() else {}
    plan = args.only or ["off", "prescreen", "gate", "replace", "live", "n8n"]
    for step in plan:
        print(f"=== {step} started {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
        if step == "n8n":
            n8n_cache("aside")
            try:
                runs["n8n"] = run_n8n(args)
                if runs["n8n"]["execution"]["status"] != "success":
                    path.write_text(json.dumps(runs, indent=2, default=str))
                    runs["n8n-retry"] = run_n8n(args, retry=True)
            finally:
                n8n_cache("restore")
        elif step == "live":
            runs["live-1"] = run_live(args, "live-1")
            runs["live-2"] = run_live(args, "live-2")
        else:
            runs[f"mira-{step}"] = run_mira(args, step)
        path.write_text(json.dumps(runs, indent=2, default=str))
        print(json.dumps({k: v for k, v in runs.items() if k.startswith(step) or k == step or
                          (step == "off" and k == "mira-off")}, default=str)[:800], flush=True)
    print(f"done → {path}")


if __name__ == "__main__":
    main()

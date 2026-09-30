#!/usr/bin/env python3
"""Turn a weekly experiment (tools/experiment_weekly.py) into one comparison:
time and cost per stage, papers kept, selection overlap.

  .venv/bin/python tools/experiment_analysis.py --current-date 2026-09-28
Writes data/experiment/analysis_<date>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from mira.paths import DATA_DIR  # noqa: E402
from n8n_replay import main_items  # noqa: E402

EXP = DATA_DIR / "experiment"

# n8n node -> stage (the same stage names as mira's ledger)
N8N_STAGES = {
    "Extract PDFs (Batch)": "first-page PDFs",
    "Note: Controlled From Profile": "credibility (affiliation)",
    "Note: Controlled From Profile1": "relevance (classification)",
    "OpenRouter Chat Model (Selection)": "selection",
    "OpenRouter Chat Model (Deep Analysis)": "deep analysis",
    "OpenRouter Chat Model2": "analysis auto-fix",
    "OpenRouter Chat Model (Report)": "report",
    "OpenRouter Chat Model5": "report re-parse",
    "OpenRouter Chat Model (Trend)": "trend",
    "Media - OpenRouter Chat Model (Selection)": "news selection + summary",
    "Media - OpenRouter Chat Model (Summarize)": "news selection + summary",
    "Download Full PDF": "full-text PDFs",
}


def bare(x) -> str:
    return re.sub(r"v\d+$", "", str(x).split("/abs/")[-1])


def n8n_side(exec_path: Path, wf: dict, prices: dict) -> dict:
    rd = json.loads(exec_path.read_text())["runData"]
    model_of = {n["name"]: n["parameters"].get("model") for n in wf["nodes"] if "lmChat" in n["type"]}
    model_of["Note: Controlled From Profile"] = model_of["Note: Controlled From Profile1"] = "openai/gpt-5.6-terra"
    stages: dict = {}
    for node, stage in N8N_STAGES.items():
        runs = rd.get(node)
        if not runs:
            continue
        span = (max(r["startTime"] + r.get("executionTime", 0) for r in runs)
                - min(r["startTime"] for r in runs)) / 1000
        cost = calls = 0
        for r in runs:
            try:
                u = r["data"]["ai_languageModel"][0][0]["json"]["tokenUsage"]
            except (KeyError, IndexError, TypeError):
                continue
            pi, po = prices.get(model_of.get(node), (0, 0))
            cost += u.get("promptTokens", 0) * pi + u.get("completionTokens", 0) * po
            calls += 1
        s = stages.setdefault(stage, {"seconds": 0.0, "cost_usd": 0.0, "calls": 0})
        s["seconds"] += span
        s["cost_usd"] += cost
        s["calls"] += calls
    pool = [bare(i.get("arxiv_id") or i.get("id")) for i in main_items(rd, "Rank and Cap Filtered Papers")] \
        if "Rank and Cap Filtered Papers" in rd else []
    selected = []
    if "Paper Selection Agent" in rd:
        out = main_items(rd, "Paper Selection Agent")[0].get("output") or {}
        selected = [bare(s.get("arxiv_id")) for s in out.get("selected_papers", [])]
    fetched = len(main_items(rd, "Remove Duplicates")) if "Remove Duplicates" in rd else None
    return {"stages": stages, "pool": pool, "selected": selected, "papers_fetched": fetched,
            "token_priced_cost_usd": round(sum(s["cost_usd"] for s in stages.values()), 3)}


def mira_side(summary_path: str) -> dict:
    s = json.loads(Path(summary_path).read_text())
    stages: dict = {}
    for r in s.get("llm", []):
        st = stages.setdefault(r["stage"], {"cost_usd": 0.0, "calls": 0, "llm_seconds": 0.0})
        st["cost_usd"] += r["cost_usd"]
        st["calls"] += int(r["calls"])
        st["llm_seconds"] += r["seconds"]
    return {"stages": stages, "stage_seconds": s.get("stage_seconds", {}),
            "pool": [bare(p["id"]) for p in s.get("ranked_pool", [])],
            "selected": [bare(p["id"]) for p in s.get("selected", [])],
            "titles": {bare(p["id"]): p["title"] for p in s.get("ranked_pool", []) + s.get("selected", [])},
            "papers_fetched": s.get("papers_fetched"), "papers_screened_by_jev": s.get("papers_screened_by_jev"),
            "llm_cost_usd": s.get("llm_cost_usd"), "jev_cost_usd": s.get("jev_cost_usd"),
            "jev": s.get("jev"), "llm_calls": s.get("llm_calls"), "jev_calls": s.get("jev_calls"),
            "subject": s.get("subject"), "wall_seconds": s.get("wall_seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-date", default="2026-09-28")
    args = ap.parse_args()
    runs = json.loads((EXP / f"weekly_{args.current_date}.json").read_text())
    ms = {m["id"]: m for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
    prices = {k: (float(v["pricing"]["prompt"]), float(v["pricing"]["completion"])) for k, v in ms.items()}
    wf = json.loads((DATA_DIR / "parity" / "workflow.json").read_text())

    out = {"runs": {}}
    if "n8n" in runs:
        r = runs["n8n"]
        ex = DATA_DIR / "parity" / f"exec_{r['execution']['id']}.json"
        side = n8n_side(ex, wf, prices) if ex.exists() else {}
        out["runs"]["n8n"] = {**side, "wall_seconds": r["wall_seconds"], "status": r["execution"]["status"],
                              "cost_measured_usd": r["cost_measured_usd"]}
    for key in ("mira-off", "mira-prescreen", "mira-gate", "mira-replace"):
        r = runs.get(key)
        if not r or not isinstance(r.get("result"), dict) or "summary_path" not in r["result"]:
            continue
        out["runs"][key] = {**mira_side(r["result"]["summary_path"]), "wall_seconds": r["wall_seconds"],
                            "cost_measured_usd": r["cost_measured_usd"], "status": r["state"]}
    for key in ("live-1", "live-2"):
        if key in runs:
            out["runs"][key] = runs[key]

    base = out["runs"].get("mira-off", {})
    n8n = out["runs"].get("n8n", {})
    for key, r in out["runs"].items():
        if "pool" not in r:
            continue
        q = {}
        if base.get("pool") and key != "mira-off":
            bp = set(base["pool"])
            q["all_llm_pool_kept"] = round(len(bp & set(r["pool"])) / len(bp), 3)
        if base.get("selected") and key != "mira-off":
            q["selected_overlap_with_all_llm"] = len(set(base["selected"]) & set(r["selected"]))
        if n8n.get("selected") and key != "n8n":
            q["selected_overlap_with_n8n"] = len(set(n8n["selected"]) & set(r["selected"]))
        if n8n.get("pool") and key != "n8n":
            a, b = set(n8n["pool"]), set(r["pool"])
            q["pool_overlap_with_n8n"] = round(len(a & b) / max(1, len(a | b)), 3)
        r["quality"] = q

    path = EXP / f"analysis_{args.current_date}.json"
    path.write_text(json.dumps(out, indent=2))
    for k, r in out["runs"].items():
        print(f"{k:16} status={r.get('status')} wall={r.get('wall_seconds')}s measured=${r.get('cost_measured_usd')} "
              f"pool={len(r.get('pool', []))} sel={len(r.get('selected', []))} q={r.get('quality')}")
    print(f"→ {path}")


if __name__ == "__main__":
    main()

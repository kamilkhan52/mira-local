#!/usr/bin/env python3
"""Calibrate a decision model's relevance-level cutoffs per profile from the
benchmark results (scripts/bench_jev.py, seeds 42 and 43), with the methods
documented above jev.CUTOFFS:

  prescreen  highest cutoff (coarse grid) keeping >= 99% of LLM-passing and
             100% of report-featured papers on both samples; None when it
             would screen out under 20% of papers
  gate       highest cutoff keeping >= 95% of LLM-passing papers (pooled)
  decide     best F1 against the LLM's pass/fail on the random papers (pooled;
             middle of the plateau when cutoffs tie)
  priority   best F1 against LLM relevance >= 7 on the random papers (pooled)
  cred_gate  agreement-maximising credibility level on first-page headers

  JEV_BACKEND=nimble .venv/bin/python scripts/calibrate_cutoffs.py
Prints the cutoff table (paste into jev.CUTOFFS / jev.LOCAL_CUTOFFS) and
writes data/report-files/bench_jev/cutoffs_<backend>.json with the stats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_jev import OUT, _to_int, load_profiles  # noqa: E402
from mira import jev  # noqa: E402

GRID = [x / 100 for x in range(0, 301)]
# The prescreen is a safety cutoff: a coarse grid, and only worth having when
# it removes a meaningful share of papers.
PRESCREEN_GRID = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
MIN_PRESCREEN_SAVING = 0.2
SAMPLES = {42: "sample_150.json", 43: "sample_150_seed43.json"}


def f1(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(t and not p for p, t in zip(pred, truth))
    return 2 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def rows_for(pid, profiles, results):
    th = profiles[pid]["thresholds"]
    rmin, cmin = th["relevance_score_min"], th["credibility_tier_min"]
    out = []
    for seed, name in SAMPLES.items():
        block = json.loads((OUT / name).read_text())[pid]
        for i, b in block["rows"].items():
            j = results.get(f"{pid}|{i}|paper")
            if not j:
                continue
            rel, cred = _to_int(b["relevance_score"]), _to_int(b["credibility_tier"])
            page = results.get(f"{pid}|{i}|cred_page")
            out.append({"seed": seed, "random": "random" in b["groups"], "featured": bool(b.get("featured")),
                        "rel": rel, "cred": cred, "passing": rel >= rmin and cred >= cmin,
                        "cred_ok": cred >= cmin, "level": j["relevance_level"],
                        "cred_level": page["credibility_level"] if page else None})
    return out


def recall(rows, c, key):
    sel = [r for r in rows if r[key]]
    return sum(r["level"] >= c for r in sel) / max(1, len(sel)), len(sel)


def calibrate(pid, rows):
    seeds = sorted({r["seed"] for r in rows})
    by_seed = {s: [r for r in rows if r["seed"] == s] for s in seeds}
    rand = [r for r in rows if r["random"]]
    res = {}

    ok = [c for c in PRESCREEN_GRID if all(recall(by_seed[s], c, "passing")[0] >= 0.99
                                 and recall(by_seed[s], c, "featured")[0] == 1.0 for s in seeds)]
    pre = max(ok) if ok else 0.0
    saved = sum(r["level"] < pre for r in rand) / max(1, len(rand))
    res["prescreen"] = pre if pre > 0 and saved >= MIN_PRESCREEN_SAVING else None

    gate = max(c for c in GRID if recall(rows, c, "passing")[0] >= 0.95)
    res["gate"] = gate

    def best_f1(truth_key):
        # middle of the best-F1 plateau (cutoffs that select the same papers tie)
        truth = [truth_key(r) for r in rand]
        scores = [(round(f1([r["level"] >= c for r in rand], truth)[0], 9), c) for c in GRID]
        top = max(sc for sc, _ in scores)
        tied = [c for sc, c in scores if sc == top]
        return tied[len(tied) // 2]

    res["decide"] = best_f1(lambda r: r["passing"])
    res["priority"] = best_f1(lambda r: r["rel"] >= 7)

    cp = [r for r in rows if r["cred_level"] is not None and r["cred"] > 0]
    if cp:
        res["cred_gate"] = max((sum((r["cred_level"] >= c) == r["cred_ok"] for r in cp), -c, c)
                               for c in GRID[:101])[2]
    else:
        res["cred_gate"] = None

    rs = sum(r["level"] < gate for r in rand) / max(1, len(rand))
    dec = f1([r["level"] >= res["decide"] for r in rand], [r["passing"] for r in rand])
    stats = {
        "n": len(rows), "random": len(rand), "cred_pages": len(cp),
        "prescreen_saved": None if res["prescreen"] is None else
        round(sum(r["level"] < res["prescreen"] for r in rand) / max(1, len(rand)), 3),
        "gate_saved": round(rs, 3),
        "gate_passing_kept": round(recall(rows, gate, "passing")[0], 3),
        "gate_featured_kept": round(recall(rows, gate, "featured")[0], 3),
        "decide_f1": round(dec[0], 3),
        "decide_precision": round(dec[1] / max(1, dec[1] + dec[2]), 3),
        "decide_recall": round(dec[1] / max(1, dec[1] + dec[3]), 3),
    }
    return res, stats


def main():
    profiles = load_profiles()
    tag = "" if jev.BACKEND == "typesafe" else f"_{jev.BACKEND}"
    results = {}
    for line in (OUT / f"results{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        results[r["key"]] = r
    table, stats = {}, {}
    for pid in json.loads((OUT / SAMPLES[42]).read_text()):
        rows = rows_for(pid, profiles, results)
        if not rows:
            continue
        table[pid], stats[pid] = calibrate(pid, rows)
    (OUT / f"cutoffs_{jev.BACKEND}.json").write_text(json.dumps({"cutoffs": table, "stats": stats}, indent=2))
    for pid in table:
        print(f'    "{pid}": {json.dumps(table[pid])},')
        print(f"        # {stats[pid]}")


if __name__ == "__main__":
    main()

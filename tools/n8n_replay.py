#!/usr/bin/env python3
"""Deterministic n8n → mira parity replay.

Takes an exported n8n execution (tools/n8n_export.py) and re-runs the mira
digest on exactly the same inputs: the same arXiv items, first-page text,
full-text downloads and crawled news, with every LLM call answered from the
recorded n8n agent outputs. Each LLM call is checked for byte-identical
prompts; stage outputs and the final email HTML are compared with n8n's.

  .venv/bin/python tools/n8n_replay.py data/parity/exec_2085.json \
      --profile memory-innovation --mode daily --current-date 2026-09-26 \
      --lookback-days 1 --test-mode

Needs the n8n cache for the profile under data/report-files/cache/ (so the
per-paper stages hit cache) — the replay stops rather than spend money if a
paper-level call has no recording.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOOL_SUFFIX = ("IMPORTANT: For your response to user, you MUST use the `format_final_json_response` tool")


def main_items(rd, node, run=0, out=0):
    r = rd[node][run]
    m = (r.get("data") or {}).get("main") or []
    return [i["json"] for i in (m[out] if len(m) > out and m[out] else [])]


def split_messages(msg: str) -> tuple[str, str]:
    """n8n LangChain message string → (system, human)."""
    system, human = "", msg
    if msg.startswith("System: "):
        i = msg.find("\nHuman: ")
        system, human = msg[len("System: "):i], msg[i + len("\nHuman: "):]
    elif msg.startswith("Human: "):
        human = msg[len("Human: "):]
    # Full system message, including n8n's tool instruction: mira sends the
    # same instruction for structured stages (mira/structured.py).
    return system, human


class Recorder:
    """Answers mira's llm_call from n8n's recorded agent outputs."""

    def __init__(self, rd: dict, workflow: dict):
        conns = workflow["connections"]
        self.recs = []
        for lm, c in conns.items():
            targets = (c.get("ai_languageModel") or [[]])[0]
            if not targets or lm not in rd:
                continue
            agent = targets[0]["node"]
            outputs = main_items(rd, agent) if agent in rd else []
            for i, run in enumerate(rd[lm]):
                msgs = run["inputOverride"]["ai_languageModel"][0][0]["json"]["messages"]
                system, human = split_messages(msgs[0])
                out = outputs[i]["output"] if i < len(outputs) and "output" in outputs[i] else None
                # n8n runs multi-item agents concurrently: LM call order is not
                # item order. Pair by the paper id the prompt mentions.
                for o in outputs:
                    oid = (o.get("output") or {}).get("arxiv_id") if isinstance(o.get("output"), dict) else None
                    if oid and len(outputs) > 1 and oid in human:
                        out = o["output"]
                        break
                self.recs.append({"lm": lm, "agent": agent, "i": i, "system": system, "human": human,
                                  "output": out, "used": False})
        self.log = []

    def __call__(self, client, model, system, user, *a, **k):
        system = system or ""
        if k.get("schema") is not None:  # what the real llm_call sends for structured stages
            from mira.structured import system_with_instruction
            system = system_with_instruction(system)
        exact = [r for r in self.recs if r["human"] == user and r["system"] == system]
        if exact:
            r = next((x for x in exact if not x["used"]), exact[0])
            status = "exact"
        else:
            r = max(self.recs, key=lambda x: difflib.SequenceMatcher(
                None, x["human"][:4000], user[:4000]).quick_ratio())
            status = "MISMATCH"
        r["used"] = True
        entry = {"agent": r["agent"], "i": r["i"], "status": status, "model": model}
        if status != "exact":
            entry["human_diff"] = "\n".join(list(difflib.unified_diff(
                r["human"].splitlines(), user.splitlines(), "n8n", "mira", n=1, lineterm=""))[:60])
            entry["system_diff"] = "\n".join(list(difflib.unified_diff(
                r["system"].splitlines(), system.splitlines(), "n8n", "mira", n=1, lineterm=""))[:30])
        self.log.append(entry)
        out = r["output"]
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def to_mira_paper(item: dict) -> dict:
    """n8n 'Remove Duplicates' item → mira.fetch paper dict."""
    def jl(v):
        if isinstance(v, list):
            return v
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else [x]
        except (TypeError, ValueError):
            return [v] if v else []
    raw = item["id"]
    return {"id": raw.split("/abs/")[-1].split("v")[0], "raw_id": raw, "title": item["title"],
            "summary": item["summary"], "published": (item.get("published") or "")[:10],
            "authors": jl(item.get("author")), "categories": jl(item.get("category")),
            "first_page_text": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exec_json")
    ap.add_argument("--profile", default="memory-innovation")
    ap.add_argument("--mode", default="daily")
    ap.add_argument("--current-date", required=True)
    ap.add_argument("--lookback-days", type=int)
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--workflow", default=str(ROOT / "data" / "parity" / "workflow.json"))
    args = ap.parse_args()

    ex = json.loads(Path(args.exec_json).read_text())
    rd = ex["runData"]
    rec = Recorder(rd, json.loads(Path(args.workflow).read_text()))

    import mira.config
    import mira.pipeline
    import mira.report
    import mira.media
    for mod in (mira.config, mira.pipeline, mira.report, mira.media):
        if hasattr(mod, "llm_call"):
            setattr(mod, "llm_call", rec)

    # Full-text downloads from n8n's 'Download Full PDF'.
    fulltext = {}
    for it in main_items(rd, "Download Full PDF"):
        try:
            res = json.loads(it.get("stdout") or "{}")
        except ValueError:
            res = {}
        if res.get("arxiv_id") or res.get("id"):
            fulltext[str(res.get("arxiv_id") or res.get("id"))] = res
    sel_ids = [i.get("selected_papers", {}).get("arxiv_id") for i in main_items(rd, "Merge Full Text Data")]
    dl_items = main_items(rd, "Download Full PDF")
    for sid, it in zip(sel_ids, dl_items):
        try:
            fulltext.setdefault(re.sub(r"v\d+$", "", sid.split("/abs/")[-1]), json.loads(it.get("stdout") or "{}"))
        except (ValueError, AttributeError):
            pass
    fake = types.ModuleType("download_full_arxiv_pdf")

    def download_and_extract(arxiv_id, *a, **k):
        key = re.sub(r"v\d+$", "", str(arxiv_id).split("/abs/")[-1])
        if key not in fulltext:
            raise RuntimeError(f"no recorded full text for {arxiv_id}")
        return fulltext[key]
    fake.download_and_extract = download_and_extract
    sys.modules["download_full_arxiv_pdf"] = fake

    from mira.config import load_config
    config = load_config(args.profile, args.mode, current_date=args.current_date,
                         lookback_days=args.lookback_days, test_mode=args.test_mode)

    papers = [to_mira_paper(i) for i in main_items(rd, "Remove Duplicates")]
    fp = {i["id"]: i.get("first_page_text", "") for i in main_items(rd, "Distribute PDF Results")}
    for p in papers:
        p["first_page_text"] = fp.get(p["raw_id"], "")

    report = {"checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'OK ' if ok else 'DIFF'}] {name}{' — ' + detail if detail and not ok else ''}")

    # arXiv parse parity: would mira's own fetch produce the same items?
    try:
        from mira.fetch import fetch_papers
        live = {p["raw_id"]: p for p in fetch_papers(config)}
        same = sum(1 for p in papers if p["raw_id"] in live and live[p["raw_id"]]["title"] == p["title"]
                   and live[p["raw_id"]]["summary"] == p["summary"])
        check("arXiv fetch/parse (ids, titles, abstracts) identical", same == len(papers),
              f"{same}/{len(papers)} identical; {len(live)} fetched live now")
    except Exception as e:  # noqa: BLE001
        check("arXiv fetch/parse", False, f"could not fetch: {e}")

    result = mira.pipeline.run_pipeline(papers, config, None)
    n8n_rank = [i.get("arxiv_id") or i.get("id") for i in main_items(rd, "Rank and Cap Filtered Papers")]
    mira_rank = [p.get("raw_id") or p.get("arxiv_id") for p in result["ranked"]]
    check("ranked pool (ids and order)", n8n_rank == mira_rank, f"n8n {len(n8n_rank)} vs mira {len(mira_rank)}")

    media_articles = []
    for node in ("Media - Map EE Times to Pipeline", "Media - Map SemiAnalysis to Pipeline",
                 "Media - Map TrendForce to Pipeline"):
        if node in rd:
            media_articles += main_items(rd, node)
    media_out = mira.media.run_media(config, None, articles=media_articles) if media_articles else None

    result["relevant_pool"] = result.get("selection_pool")
    n8n_persist = main_items(rd, "Persist Current Report")[0]
    created = n8n_persist["report_record"].get("created_at")
    now = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc)
    out = mira.report.produce_report(result, media_out, config, None, now=now, pdf=False)

    n8n_combined = main_items(rd, "Combine Report Data")[0]
    from mira.report import build_report_inputs
    combined = build_report_inputs(result, media_out, config)["combined"]
    for key in ("stats", "stats_dashboard", "this_period_numbers", "media_intelligence"):
        a, b = n8n_combined.get(key), combined.get(key)
        check(f"Combine Report Data: {key}", a == b,
              "" if a == b else json.dumps({"n8n": a, "mira": b}, default=str)[:600])

    n8n_style = main_items(rd, "Apply Email Styling")[0]
    check("subject", n8n_style["subject"] == out["subject"], f"n8n={n8n_style['subject']!r} mira={out['subject']!r}")
    same_html = n8n_style["html_body"] == out["html"]
    diff = ""
    if not same_html:
        a = re.sub(r">\s*<", ">\n<", n8n_style["html_body"]).splitlines()
        b = re.sub(r">\s*<", ">\n<", out["html"]).splitlines()
        diff = "\n".join(list(difflib.unified_diff(a, b, "n8n", "mira", n=1, lineterm=""))[:200])
    check("final email HTML byte-identical", same_html, f"{len(diff.splitlines())} diff lines")

    llm_ok = [e for e in rec.log if e["status"] == "exact"]
    check("every LLM prompt byte-identical to n8n's", len(llm_ok) == len(rec.log),
          f"{len(llm_ok)}/{len(rec.log)} exact")
    unused = [f"{r['agent']}#{r['i']}" for r in rec.recs if not r["used"]]
    check("every n8n LLM call reproduced", not unused, f"not reproduced: {unused}")

    outdir = ROOT / "data" / "parity"
    stem = Path(args.exec_json).stem
    (outdir / f"{stem}_mira.html").write_text(out["html"])
    (outdir / f"{stem}_n8n.html").write_text(n8n_style["html_body"])
    (outdir / f"{stem}_replay.json").write_text(json.dumps(
        {"checks": report["checks"], "llm_calls": rec.log, "html_diff": diff}, indent=2, ensure_ascii=False))
    print(f"\nDetails: {outdir / (stem + '_replay.json')}")


if __name__ == "__main__":
    main()

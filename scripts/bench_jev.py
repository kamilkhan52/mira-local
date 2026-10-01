#!/usr/bin/env python3
"""Benchmark Jev against the LLM baseline on the per-paper triage stages.

Baseline = the cached LLM results the workflows already produced
(report-files/cache/<profile>/{classification,affiliation}). For a stratified
sample per profile this script fetches the same inputs the LLM saw (arXiv
title/abstract, first page), runs Jev, and compares:

  relevance      Jev Score vs LLM relevance_score (rank corr, gate agreement)
  cascade        Jev as a pre-screen: how many LLM calls a cutoff saves vs how
                 many LLM-passing / report-featured papers it would drop
  primary_topic  Jev Choice vs LLM label (where the label maps to the taxonomy)
  impact/actionable  label agreement
  credibility    Jev Score from (a) the LLM-extracted affiliation list and
                 (b) the raw first-page header, vs LLM credibility_tier
  latency        Jev per call; optional live LLM timing (--llm-timing N)

Everything is cached under report-files/bench_jev/ so reruns only compute
what is missing. Usage:

  python3.11 scripts/bench_jev.py --per-profile 150 --first-pages 50 --llm-timing 40
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mira import jev  # noqa: E402
from mira.paths import CONFIG_DIR, REPORT_FILES  # noqa: E402

OUT = REPORT_FILES / "bench_jev"
PROFILES = ["memory-innovation", "cxl-research", "storage-innovation", "optical-interconnects"]
CACHE = REPORT_FILES / "cache"


def bare(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.split("/abs/")[-1])


def _to_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_profiles() -> dict[str, dict]:
    d = json.loads((CONFIG_DIR / "memory-innovation-profile.json").read_text())
    return {p["profile_id"]: p for p in d["profiles"]}


def load_baseline(profile_id: str) -> dict[str, dict]:
    """Latest cached LLM result per paper, classification joined with affiliation."""
    out: dict[str, dict] = {}
    for stage in ("classification", "affiliation"):
        latest: dict[str, dict] = {}
        for f in glob.glob(str(CACHE / profile_id / stage / "*.json")):
            try:
                e = json.loads(Path(f).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            res = e.get("result")
            if not isinstance(res, dict) or not e.get("arxiv_id"):
                continue
            k = bare(e["arxiv_id"])
            if k not in latest or e.get("created_at", "") > latest[k].get("created_at", ""):
                latest[k] = e
        for k, e in latest.items():
            row = out.setdefault(k, {"id": k})
            row[f"{stage}_model"] = e.get("model")
            row.update(e["result"])
    return {k: v for k, v in out.items()
            if "relevance_score" in v and "credibility_tier" in v}


def featured_ids() -> dict[str, set[str]]:
    ids: dict[str, set[str]] = defaultdict(set)
    for f in glob.glob(str(REPORT_FILES / "prod" / "*" / "*.json")):
        prof = Path(f).parent.name
        ids[prof] |= set(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", Path(f).read_text()))
    return ids


def build_sample(per_profile: int, profiles: dict, seed: int = 42, exclude: set | None = None) -> dict:
    path = OUT / (f"sample_{per_profile}.json" if seed == 42 else f"sample_{per_profile}_seed{seed}.json")
    if path.exists():
        return json.loads(path.read_text())
    feats = featured_ids()
    rng = random.Random(seed)
    exclude = exclude or set()
    sample = {}
    for pid in PROFILES:
        base = load_baseline(pid)
        th = profiles[pid]["thresholds"]
        ids = sorted(i for i in base if i not in exclude)
        passing = [i for i in ids if _to_int(base[i]["relevance_score"]) >= th["relevance_score_min"]
                   and _to_int(base[i]["credibility_tier"]) >= th["credibility_tier_min"]]
        feat = [i for i in ids if i in feats.get(pid, set())]
        rand = rng.sample(ids, min(per_profile, len(ids)))
        extra_pass = rng.sample([i for i in passing if i not in rand], min(per_profile // 2, len(passing)))
        extra_feat = rng.sample([i for i in feat if i not in rand], min(per_profile // 2, len(feat)))
        rows = {}
        for group, grp_ids in (("random", rand), ("passing", extra_pass), ("featured", extra_feat)):
            for i in grp_ids:
                r = rows.setdefault(i, dict(base[i]))
                r.setdefault("groups", []).append(group)
        for i, r in rows.items():
            r["featured"] = i in feats.get(pid, set())
        sample[pid] = {"rows": rows, "population": len(ids),
                       "population_pass_rate": len(passing) / max(1, len(ids))}
        print(f"  {pid}: population {len(ids)}, pass rate {len(passing)/max(1,len(ids)):.1%}, "
              f"featured-with-baseline {len(feat)}, sampled {len(rows)}")
    path.write_text(json.dumps(sample))
    return sample


def fetch_metadata(ids: list[str]) -> dict[str, dict]:
    """arXiv title/abstract/authors for bare ids, cached."""
    from mira.fetch import _parse_xml, _HEADERS
    path = OUT / "metadata.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    todo = [i for i in dict.fromkeys(ids) if i not in meta]
    for n in range(0, len(todo), 100):
        batch = todo[n:n + 100]
        for attempt in range(4):
            try:
                r = requests.get("https://export.arxiv.org/api/query",
                                 params={"id_list": ",".join(batch), "max_results": len(batch)},
                                 headers=_HEADERS, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"    arXiv retry {attempt + 1}: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for p in _parse_xml(r.text):
            meta[p["id"]] = p
        print(f"    metadata {min(n + 100, len(todo))}/{len(todo)}")
        path.write_text(json.dumps(meta))
        time.sleep(3.1)
    return meta


def fetch_first_pages(papers: list[dict]) -> dict[str, str]:
    from mira.fetch import extract_first_pages
    path = OUT / "first_pages.json"
    pages = json.loads(path.read_text()) if path.exists() else {}
    todo = [dict(p) for p in papers if p["id"] not in pages]
    if todo:
        for p in extract_first_pages(todo):
            if p.get("first_page_text", "").strip():
                pages[p["id"]] = p["first_page_text"]
        path.write_text(json.dumps(pages))
    return pages


class ResultStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[r["key"]] = r
        self.fh = path.open("a")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        value = {"key": key, **value}
        self.data[key] = value
        self.fh.write(json.dumps(value) + "\n")
        self.fh.flush()


def run_jev(sample, profiles, meta, pages, store: ResultStore, workers: int):
    tasks = []
    for pid, block in sample.items():
        rubric = jev.profile_rubric(profiles[pid])
        focus = profiles[pid]["topic"]["focus"]
        for i, row in block["rows"].items():
            m = meta.get(i)
            if not m:
                continue
            local = jev.BACKEND != "typesafe"
            judge = jev.judge_relevance if local else jev.judge_paper
            if not store.get(f"{pid}|{i}|paper"):
                tasks.append((f"{pid}|{i}|paper", lambda m=m, r=rubric, j=judge: j(m, r)))
            affs = row.get("affiliations")
            # Local backends: skip credibility-from-affiliation-list (no level uses it).
            if not local and isinstance(affs, list) and not store.get(f"{pid}|{i}|cred_aff"):
                affs = [a for a in affs if a and a != "Unknown"]
                tasks.append((f"{pid}|{i}|cred_aff",
                              lambda f=focus, a=affs: jev.judge_credibility(f, affiliations=a)))
            if i in pages and not store.get(f"{pid}|{i}|cred_page"):
                tasks.append((f"{pid}|{i}|cred_page",
                              lambda f=focus, t=pages[i]: jev.judge_credibility(f, first_page_text=t)))
    print(f"  Jev calls to make: {len(tasks)}")

    def go(t):
        key, fn = t
        try:
            store.put(key, fn())
        except Exception as e:  # noqa: BLE001 — record and continue
            print(f"    Jev error {key}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(go, tasks), 1):
            if n % 200 == 0:
                print(f"    {n}/{len(tasks)}")


def run_llm_timing(sample, profiles, meta, store: ResultStore, n: int):
    """Fresh LLM classification with the production model/prompt for n papers of
    the memory profile: gives wall-clock latency and an LLM-vs-LLM consistency
    reference (current model vs cached baseline)."""
    from mira.config import llm_call, make_llm_client, parse_json_response
    from mira.pipeline import _apply_template_n8n

    raw = json.loads((CONFIG_DIR / "memory-innovation-profile.json").read_text())
    pid = "memory-innovation"
    prof = profiles[pid]
    model = raw["llm_models"]["classification"]
    rows = [i for i, r in sample[pid]["rows"].items() if "random" in r["groups"] and i in meta]
    rows = rows[:n]
    client = make_llm_client()
    system = _apply_template_n8n(prof["prompts"]["classification"].get("system", ""),
                                 {"topic_focus": prof["topic"]["focus"]})

    def go(i):
        key = f"{pid}|{i}|llm_cls"
        if store.get(key):
            return
        m = meta[i]
        user = _apply_template_n8n(prof["prompts"]["classification"]["user"], {
            "arxiv_id": m["raw_id"], "title": m["title"], "summary": m["summary"]})
        t0 = time.perf_counter()
        try:
            out = llm_call(client, model, system, user, reasoning_effort="high")
        except Exception as e:  # noqa: BLE001
            print(f"    LLM error {i}: {e}")
            return
        latency = time.perf_counter() - t0
        try:
            res = parse_json_response(out)
        except Exception:  # noqa: BLE001
            res = {}
        store.put(key, {"model": model, "latency_s": latency,
                        "relevance_score": res.get("relevance_score"),
                        "primary_topic": res.get("primary_topic")})

    print(f"  LLM timing calls: {len(rows)} with {model}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(go, rows))


# ---------------------------------------------------------------- metrics --

def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def gate_stats(pred: list[bool], truth: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(t and not p for p, t in zip(pred, truth))
    tn = sum((not p) and (not t) for p, t in zip(pred, truth))
    n = len(pred) or 1
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    return {"n": len(pred), "agreement": po,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}


def _stem(label: str) -> str:
    return re.sub(r"\s*\(.*", "", str(label)).strip().lower()


def topic_match(llm_label: str, taxonomy: list[str]) -> str | None:
    """Map a free LLM label onto the taxonomy, or None if it doesn't map cleanly."""
    s = _stem(llm_label)
    if not s:
        return None
    if s.startswith("not related"):
        return next((t for t in taxonomy if t.lower().startswith("not related")), None)
    for t in taxonomy:
        ts = _stem(t)
        if s == ts or s.startswith(ts) or ts.startswith(s):
            return t
    return None


def lead_word(v) -> str:
    m = re.match(r"\s*([A-Za-z]+)", str(v or ""))
    return m.group(1).capitalize() if m else ""


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def analyse(sample, profiles, store: ResultStore) -> dict:
    report = {"profiles": {}}
    lat_paper, lat_cred = [], []
    for pid, block in sample.items():
        th = profiles[pid]["thresholds"]
        rmin, cmin = th["relevance_score_min"], th["credibility_tier_min"]
        taxonomy = jev.profile_rubric(profiles[pid])["taxonomy"]
        rows = []
        for i, b in block["rows"].items():
            j = store.get(f"{pid}|{i}|paper")
            if not j:
                continue
            rows.append((i, b, j, store.get(f"{pid}|{i}|cred_aff"), store.get(f"{pid}|{i}|cred_page")))
            lat_paper.append(j["latency_s"])
        P = {}
        llm_rel = [_to_int(b["relevance_score"]) for _, b, *_ in rows]
        jev_rel = [j["relevance_level"] for _, _, j, *_ in rows]
        P["n"] = len(rows)
        P["relevance_spearman"] = spearman(jev_rel, llm_rel)
        P["relevance_gate"] = gate_stats([j["relevance_score"] >= rmin for _, _, j, *_ in rows],
                                         [s >= rmin for s in llm_rel])
        # Relevance gate with a cutoff tuned on this data (agreement-maximising).
        best = max((sum((lv >= c) == (s >= rmin) for lv, s in zip(jev_rel, llm_rel)), c)
                   for c in [x / 20 for x in range(0, 81)])
        P["relevance_gate_tuned"] = {"cutoff_level": best[1], **gate_stats(
            [lv >= best[1] for lv in jev_rel], [s >= rmin for s in llm_rel])}

        # Cascade: Jev screens out papers below a cutoff; the rest go to the LLM
        # unchanged. Savings measured on the random group (population estimate);
        # misses measured on all LLM-passing and all report-featured papers.
        rand = [(j["relevance_level"]) for _, b, j, *_ in rows if "random" in b["groups"]]
        passing = [j["relevance_level"] for _, b, j, *_ in rows
                   if _to_int(b["relevance_score"]) >= rmin and _to_int(b["credibility_tier"]) >= cmin]
        feat = [j["relevance_level"] for _, b, j, *_ in rows if b.get("featured")]
        casc = []
        for c in sorted({x / 100 for x in range(0, 31)} | {x / 10 for x in range(0, 31)}):
            casc.append({"cutoff_level": c,
                         "llm_calls_saved": sum(v < c for v in rand) / max(1, len(rand)),
                         "passing_recall": sum(v >= c for v in passing) / max(1, len(passing)),
                         "featured_recall": sum(v >= c for v in feat) / max(1, len(feat))})
        P["cascade"] = casc
        P["cascade_counts"] = {"random": len(rand), "passing": len(passing), "featured": len(feat)}

        # Topic
        # Topic agreement only where the LLM passed the paper: for off-topic
        # papers the profiles' taxonomies disagree on how to say "not related"
        # (cxl/optical have no such category), so agreement there is not meaningful.
        mapped = [(topic_match(b.get("primary_topic", ""), taxonomy), j["primary_topic"])
                  for _, b, j, *_ in rows if _to_int(b["relevance_score"]) >= rmin]
        mapped_all = [(topic_match(b.get("primary_topic", ""), taxonomy), j["primary_topic"])
                      for _, b, j, *_ in rows]
        m_ok = [(a, c) for a, c in mapped if a]
        P["topic_mappable"] = len(m_ok) / max(1, len(mapped))
        P["topic_agreement"] = sum(a == c for a, c in m_ok) / max(1, len(m_ok))
        nr = [(a.lower().startswith("not related"), c.lower().startswith("not related"))
              for a, c in mapped_all if a]
        P["not_related_gate"] = gate_stats([c for _, c in nr], [a for a, _ in nr])
        # Impact / actionable
        imp = [(lead_word(b.get("potential_impact")), j["potential_impact"]) for _, b, j, *_ in rows]
        imp = [(a, c) for a, c in imp if a in jev.IMPACT_OPTIONS]
        P["impact_agreement"] = sum(a == c for a, c in imp) / max(1, len(imp))
        order = list(jev.IMPACT_OPTIONS)
        P["impact_within_one"] = sum(abs(order.index(a) - order.index(c)) <= 1 for a, c in imp) / max(1, len(imp))
        act = [(lead_word(b.get("actionable")), j["actionable"]) for _, b, j, *_ in rows]
        act = [(a, c) for a, c in act if a in jev.ACTIONABLE_OPTIONS]
        P["actionable_agreement"] = sum(a == c for a, c in act) / max(1, len(act))

        # Credibility
        for kind, idx in (("cred_aff", 3), ("cred_page", 4)):
            pairs = [(r[idx], _to_int(r[1]["credibility_tier"])) for r in rows if r[idx]]
            pairs = [(c, t) for c, t in pairs if t > 0]
            lat_cred.extend(c["latency_s"] for c, _ in pairs)
            if not pairs:
                continue
            P[kind] = {
                "n": len(pairs),
                "spearman": spearman([c["credibility_level"] for c, _ in pairs], [t for _, t in pairs]),
                "mean_abs_diff_tier": statistics.mean(abs(c["credibility_tier"] - t) for c, t in pairs),
                "gate": gate_stats([c["credibility_tier"] >= cmin for c, _ in pairs],
                                   [t >= cmin for _, t in pairs]),
            }
        report["profiles"][pid] = P

    report["latency"] = {
        "jev_paper_p50_s": pct(lat_paper, .5), "jev_paper_p95_s": pct(lat_paper, .95),
        "jev_cred_p50_s": pct(lat_cred, .5), "jev_cred_p95_s": pct(lat_cred, .95),
    }
    llm = [r for k, r in store.data.items() if k.endswith("|llm_cls")]
    if llm:
        report["latency"]["llm_cls_p50_s"] = pct([r["latency_s"] for r in llm], .5)
        report["latency"]["llm_cls_p95_s"] = pct([r["latency_s"] for r in llm], .95)
        report["latency"]["llm_model"] = llm[0]["model"]
        # LLM-vs-LLM reference: current model vs cached baseline, same papers.
        base = sample["memory-innovation"]["rows"]
        pairs = [(_to_int(r["relevance_score"]), _to_int(base[r["key"].split("|")[1]]["relevance_score"]))
                 for r in llm if r.get("relevance_score") is not None]
        jpairs = [(store.get(f"memory-innovation|{r['key'].split('|')[1]}|paper") or {}).get("relevance_level")
                  for r in llm if r.get("relevance_score") is not None]
        rmin = profiles["memory-innovation"]["thresholds"]["relevance_score_min"]
        report["llm_reference"] = {
            "n": len(pairs),
            "llm_vs_baseline_spearman": spearman([a for a, _ in pairs], [b for _, b in pairs]),
            "llm_vs_baseline_gate": gate_stats([a >= rmin for a, _ in pairs], [b >= rmin for _, b in pairs]),
            "jev_vs_baseline_spearman_same_papers": spearman(
                [x for x in jpairs if x is not None],
                [b for (_, b), x in zip(pairs, jpairs) if x is not None]),
        }
    return report


def to_markdown(rep: dict, profiles: dict) -> str:
    f = lambda v: "n/a" if v != v else f"{v:.2f}"  # noqa: E731
    L = ["# Jev vs LLM baseline — per-paper triage", ""]
    lat = rep["latency"]
    L += ["## Latency", "",
          f"- Jev paper judgment (relevance+topic+impact+actionable, 1 request): "
          f"p50 {f(lat['jev_paper_p50_s'])}s, p95 {f(lat['jev_paper_p95_s'])}s",
          f"- Jev credibility: p50 {f(lat['jev_cred_p50_s'])}s, p95 {f(lat['jev_cred_p95_s'])}s"]
    if "llm_cls_p50_s" in lat:
        L.append(f"- LLM classification ({lat['llm_model']}, reasoning high): "
                 f"p50 {f(lat['llm_cls_p50_s'])}s, p95 {f(lat['llm_cls_p95_s'])}s")
    if "llm_reference" in rep:
        r = rep["llm_reference"]
        L += ["", "## Reference: LLM vs its own cached baseline (memory profile)", "",
              f"n={r['n']}: relevance Spearman {f(r['llm_vs_baseline_spearman'])}, "
              f"gate agreement {f(r['llm_vs_baseline_gate']['agreement'])}, "
              f"kappa {f(r['llm_vs_baseline_gate']['kappa'])}. "
              f"Jev on the same papers: Spearman {f(r['jev_vs_baseline_spearman_same_papers'])}."]
    L += ["", "## Per profile", "",
          "| profile | n | rel ρ | gate agree | gate κ | tuned cutoff → agree/κ | topic agree, LLM-relevant (mappable) | "
          "not-related κ | impact ±1 | actionable | cred(aff) ρ / gate κ | cred(page) ρ / gate κ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pid, P in rep["profiles"].items():
        g, t = P["relevance_gate"], P["relevance_gate_tuned"]
        ca, cp = P.get("cred_aff"), P.get("cred_page")
        L.append(
            f"| {pid} | {P['n']} | {f(P['relevance_spearman'])} | {f(g['agreement'])} | {f(g['kappa'])} | "
            f"{t['cutoff_level']:.2f} → {f(t['agreement'])}/{f(t['kappa'])} | "
            f"{f(P['topic_agreement'])} ({P['topic_mappable']:.0%}) | {f(P['not_related_gate']['kappa'])} | "
            f"{f(P['impact_within_one'])} | {f(P['actionable_agreement'])} | "
            f"{(f(ca['spearman']) + ' / ' + f(ca['gate']['kappa'])) if ca else 'n/a'} | "
            f"{(f(cp['spearman']) + ' / ' + f(cp['gate']['kappa'])) if cp else 'n/a'} |")
    L += ["", "## Cascade (Jev pre-screen, LLM unchanged for survivors)", "",
          "Share of LLM calls saved vs share of LLM-passing and report-featured papers kept.", ""]
    for pid, P in rep["profiles"].items():
        c = P["cascade_counts"]
        L += [f"### {pid} (random {c['random']}, passing {c['passing']}, featured {c['featured']})", "",
              "| cutoff level | LLM calls saved | passing kept | featured kept |", "|---|---|---|---|"]
        for row in P["cascade"]:
            if row["cutoff_level"] in (0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
                L.append(f"| {row['cutoff_level']:.2f} | {row['llm_calls_saved']:.0%} | "
                         f"{row['passing_recall']:.1%} | {row['featured_recall']:.1%} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-profile", type=int, default=150)
    ap.add_argument("--first-pages", type=int, default=50,
                    help="papers per profile to download first pages for (credibility from page header)")
    ap.add_argument("--llm-timing", type=int, default=0,
                    help="run the production classification LLM on N memory-profile papers")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42,
                    help="non-default seed draws a holdout sample disjoint from the seed-42 sample")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()

    print("Sampling baseline...")
    exclude = set()
    if args.seed != 42:
        base_sample = build_sample(args.per_profile, profiles)
        exclude = {i for b in base_sample.values() for i in b["rows"]}
    sample = build_sample(args.per_profile, profiles, args.seed, exclude)
    # One result file per decision-model backend (JEV_BACKEND), so local
    # alternatives are scored on exactly the same samples as Jev.
    tag = "" if jev.BACKEND == "typesafe" else f"_{jev.BACKEND}"
    store = ResultStore(OUT / f"results{tag}.jsonl")
    if not args.analyse_only:
        all_ids = [i for b in sample.values() for i in b["rows"]]
        print(f"Fetching arXiv metadata for {len(set(all_ids))} papers...")
        meta = fetch_metadata(all_ids)
        rng = random.Random(7)
        page_targets = []
        for b in sample.values():
            ids = [i for i in sorted(b["rows"]) if i in meta]
            page_targets += [meta[i] for i in rng.sample(ids, min(args.first_pages, len(ids)))]
        print(f"Fetching first pages for {len(page_targets)} papers...")
        pages = fetch_first_pages(page_targets)
        print("Running Jev...")
        run_jev(sample, profiles, meta, pages, store, args.workers)
        if args.llm_timing:
            run_llm_timing(sample, profiles, meta, store, args.llm_timing)

    rep = analyse(sample, profiles, store)
    suffix = ("" if args.seed == 42 else f"_seed{args.seed}") + tag
    (OUT / f"report{suffix}.json").write_text(json.dumps(rep, indent=2))
    md = to_markdown(rep, profiles)
    (OUT / f"report{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

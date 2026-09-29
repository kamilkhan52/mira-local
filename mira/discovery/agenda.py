"""Markdown agenda rendering (spec §4). The appendix carries the audit trail:
topic merge map, suppressed pairs, near-misses, per-stage counts."""
from __future__ import annotations

import re

from mira.hypothesis.gaps import GapCandidate


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _novelty(c: GapCandidate) -> str:
    if c.novelty_hits is None:
        return c.novelty_label
    return f"{c.novelty_label} ({c.novelty_hits} ext.)"


def render_agenda(
    meta: dict,
    items: list[dict],
    pursuing: list[dict],
    suppressed: list[tuple[str, str]],
    merge_map: dict[str, str],
    near_misses: list[GapCandidate],
    stage_counts: dict[str, int],
) -> str:
    lines = [
        f"# Discovery Agenda — {meta['profile']}",
        "",
        f"- **Run:** {meta['run_stamp']}",
        f"- **Graph snapshot:** {meta['graph_stats']}",
        f"- **Flags:** {meta['flags'] or '(none)'}",
        f"- **Ledger:** {meta['ledger_summary']}",
    ]
    for note in meta.get("degraded", []):
        lines.append(f"- **Degraded:** {note}")
    lines.append("")

    lines += ["## Ranked opportunities", ""]
    if not items:
        lines += ["No opportunities survived filtering. See the appendix for "
                  "all mined candidates.", ""]
    else:
        lines += ["| # | Pair | Final | Combined | Novelty | Feasibility | History | Dossier |",
                  "|---|---|---|---|---|---|---|---|"]
        for i, item in enumerate(items, 1):
            c: GapCandidate = item["candidate"]
            lines.append(
                f"| {i} | {_cell(c.topic_a)} × {_cell(c.topic_c)} "
                f"| {item['final_score']:.2f} | {c.combined_score:.2f} "
                f"| {_cell(_novelty(c))} | {item['tier']} "
                f"| {_cell(item['ledger_note'])} "
                f"| {_cell(item['dossier_file'] or '—')} |"
            )
        lines += ["", "### Feasibility notes", ""]
        for item in items:
            c = item["candidate"]
            lines.append(f"- **{c.topic_a} × {c.topic_c}** [{item['tier']}] — "
                         f"{item['tier_reason']}")
        lines.append("")
        objections = [item["candidate"] for item in items
                      if item["candidate"].critic_objection]
        if objections:
            lines += ["### Critic objections", ""]
            for c in objections:
                lines += [f"**{c.topic_a} × {c.topic_c}** — {c.critic_objection}", ""]

    if pursuing:
        lines += ["## Active work (status: pursuing)", ""]
        for e in pursuing:
            lines.append(f"- **{e['key']}** — first {e['first_recommended']}, "
                         f"last {e['last_recommended']}, best score {e['best_score']:.2f}")
        lines.append("")

    lines += ["## Appendix — audit trail", ""]
    if stage_counts:
        lines += ["### Pipeline counts", ""]
        lines += [f"- {stage}: {n}" for stage, n in stage_counts.items()]
        lines.append("")
    if merge_map:
        lines += ["### Topic merge map (in-memory only)", ""]
        lines += [f"- {alias} → {canonical}" for alias, canonical in sorted(merge_map.items())]
        lines.append("")
    if suppressed:
        lines += ["### Suppressed by ledger status", ""]
        lines += [f"- {key} ({status})" for key, status in suppressed]
        lines.append("")
    if near_misses:
        lines += ["### Near-misses (below the cut or dropped by novelty)", "",
                  "| Pair | Combined | Novelty |", "|---|---|---|"]
        for c in near_misses:
            lines.append(f"| {_cell(c.topic_a)} × {_cell(c.topic_c)} "
                         f"| {c.combined_score:.2f} | {_cell(_novelty(c))} |")
        lines.append("")
    return "\n".join(lines)

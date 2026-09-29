"""Markdown dossier rendering. The appendix lists every mined candidate —
including dropped ones and why — so novelty claims are auditable (spec §5)."""
from __future__ import annotations

from .gaps import GapCandidate, gap_facts


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _novelty_line(c: GapCandidate) -> str:
    if c.novelty_hits is None:
        return f"novelty: {c.novelty_label}"
    return f"novelty: {c.novelty_label} ({c.novelty_hits} external papers, last 3 years)"


def render_dossier(meta: dict, hypotheses: list[dict], all_candidates: list[GapCandidate]) -> str:
    profiles = meta.get("profiles", meta.get("profile", ()))
    if isinstance(profiles, str):
        profiles = (profiles,)
    lines = [
        f"# Hypothesis Dossier — {meta['topic']}",
        "",
        f"- **Profiles:** {', '.join(profiles)}",
        f"- **Run date:** {meta['run_date']}",
        f"- **Graph snapshot:** {meta['graph_stats']}",
        f"- **Flags:** {meta['flags'] or '(none)'}",
        f"- **Anchor topics:** {', '.join(meta['anchors']) or '(none)'}",
    ]
    for note in meta.get("degraded", []):
        lines.append(f"- **Degraded:** {note}")
    lines.append("")

    if not hypotheses:
        lines += ["## Hypotheses", "", "No hypotheses survived filtering. "
                  "See the appendix for all mined candidates.", ""]
    else:
        lines += ["## Hypotheses", ""]
        for i, h in enumerate(hypotheses, 1):
            c: GapCandidate = h["candidate"]
            lines += [
                f"### {i}. {c.topic_a} × {c.topic_c}",
                "",
                f"*score {c.combined_score:.2f} · {_novelty_line(c)}*",
                "",
                gap_facts(c),
                "",
            ]
            if c.external_titles:
                lines += ["Closest external work: " + "; ".join(t.replace("\n", " ") for t in c.external_titles), ""]
            lines += [h["text"], ""]
            if h.get("critic_verdict"):
                lines += [f"**Critic verdict:** {h['critic_verdict']}", ""]
                if h.get("critic_body"):
                    lines += [h["critic_body"], ""]

    lines += ["## Appendix — all mined candidates", ""]
    if not all_candidates:
        lines += ["(no candidates mined)", ""]
    else:
        lines += ["| Pair | Combined | Structural | Semantic | Novelty |",
                  "|---|---|---|---|---|"]
        for c in sorted(all_candidates, key=lambda x: x.combined_score, reverse=True):
            lines.append(
                f"| {_cell(c.topic_a)} × {_cell(c.topic_c)} | {c.combined_score:.2f} "
                f"| {c.structural_score:.1f} | {c.semantic_score:.2f} "
                f"| {_cell(_novelty_line(c))} |"
            )
        lines.append("")

        objections = [c for c in all_candidates if c.critic_objection]
        if objections:
            lines += ["### Critic objections", ""]
            for c in objections:
                lines += [f"**{c.topic_a} × {c.topic_c}** — {c.critic_objection}", ""]
    return "\n".join(lines)

from mira.hypothesis.dossier import render_dossier
from mira.hypothesis.gaps import GapCandidate


def _cand(c="PIM", label="open gap", hits=1, score=0.7) -> GapCandidate:
    cand = GapCandidate("CXL", c, ["HBM4"], ["Samsung"], ["Kim"], 0.0, 10.0)
    cand.combined_score = score
    cand.novelty_label = label
    cand.novelty_hits = hits
    cand.external_titles = ["External neighbor paper"]
    return cand


def _meta() -> dict:
    return {
        "topic": "CXL memory pooling", "profile": "memory-innovation",
        "run_date": "2026-07-09", "graph_stats": "666 papers · 428 topics",
        "flags": "--critic", "anchors": ["CXL"], "degraded": [],
    }


def test_dossier_contains_header_hypotheses_and_appendix():
    dropped = _cand(c="DDR6", label="dropped", hits=42, score=0.3)
    md = render_dossier(
        _meta(),
        [{"candidate": _cand(), "text": "**Claim** — something falsifiable.",
          "critic_verdict": "strengthen", "critic_body": "Revised text."}],
        [_cand(), dropped],
    )
    assert "CXL memory pooling" in md and "memory-innovation" in md
    assert "666 papers" in md
    assert "**Claim** — something falsifiable." in md
    assert "open gap" in md and "1 external" in md
    assert "External neighbor paper" in md
    # appendix lists every mined candidate including dropped ones, with reason
    assert "DDR6" in md and "dropped" in md and "42" in md
    # critic outcome surfaces
    assert "strengthen" in md.lower()


def test_dossier_notes_degraded_mode_and_empty_results():
    meta = _meta() | {"degraded": ["Semantic Scholar unavailable — novelty unverified"]}
    md = render_dossier(meta, [], [])
    assert "Semantic Scholar unavailable" in md
    assert "No hypotheses" in md


def test_dossier_escapes_pipes_and_renders_unverified_candidates():
    weird = _cand(c="PIM | near-memory", label="unverified", hits=None, score=0.5)
    meta = _meta() | {"degraded": ["Semantic Scholar unavailable — novelty unverified"]}
    md = render_dossier(meta, [], [weird])
    # Table row must stay intact: the pipe inside the topic name is escaped.
    row = next(l for l in md.splitlines() if "PIM" in l and l.startswith("|"))
    assert "PIM \\| near-memory" in row
    assert row.count(" | ") == 4  # 5 columns -> 4 internal separators
    # Unverified novelty renders without a hit count.
    assert "novelty: unverified" in row
    assert "None external" not in md


def test_unverified_killed_candidate_keeps_audit_trail():
    killed = _cand(label="unverified · killed by critic", hits=None)
    killed.novelty_hits = None
    md = render_dossier(_meta(), [], [killed])
    row = next(l for l in md.splitlines() if "PIM" in l and l.startswith("|"))
    assert "killed by critic" in row
    assert "novelty: unverified" in row


def test_killed_candidate_objection_recorded_in_appendix():
    killed = _cand(label="open gap · killed by critic")
    killed.critic_objection = "The claim contradicts Paper P3's measured latency."
    md = render_dossier(_meta(), [], [killed])
    assert "### Critic objections" in md
    assert "CXL × PIM" in md
    assert "contradicts Paper P3" in md


def test_no_objections_section_when_none_killed():
    md = render_dossier(_meta(), [], [_cand()])
    assert "Critic objections" not in md

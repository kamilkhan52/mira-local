from mira.discovery.agenda import render_agenda, slug
from mira.hypothesis.gaps import GapCandidate


def _cand(a="CXL", c="PIM") -> GapCandidate:
    cand = GapCandidate(a, c, ["HBM4"], ["Samsung"], [], 1.0, 10.0)
    cand.combined_score = 0.83
    cand.novelty_hits = 2
    cand.novelty_label = "open gap"
    return cand


def _meta() -> dict:
    return {"profile": "memory-innovation", "run_stamp": "2026-07-13-1430",
            "graph_stats": "666 papers · 428 topics (401 after merge) · 512 in profile",
            "flags": "(none)", "ledger_summary": "3 tracked (2 new, 1 pursuing)",
            "degraded": []}


def _item(**over) -> dict:
    item = {"candidate": _cand(), "final_score": 0.75, "tier": "T2",
            "tier_reason": "Commodity CXL hardware suffices.",
            "ledger_note": "new", "dossier_file": "discovery-cxl--pim-2026-07-13-1430.md"}
    item.update(over)
    return item


def test_slug():
    assert slug("CXL memory pooling!") == "cxl-memory-pooling"


def test_agenda_renders_critic_objections():
    killed = _cand()
    killed.novelty_label += " · killed by critic"
    killed.critic_objection = "The claim contradicts the retention data."
    text = render_agenda(_meta(), [_item(candidate=killed, dossier_file=None)],
                         [], [], {}, [], {})
    assert "### Critic objections" in text
    assert "The claim contradicts the retention data." in text


def test_agenda_omits_critic_section_when_no_objections():
    text = render_agenda(_meta(), [_item()], [], [], {}, [], {})
    assert "Critic objections" not in text


def test_agenda_contains_header_table_and_sections():
    text = render_agenda(_meta(), [_item()], pursuing=[], suppressed=[],
                         merge_map={"cxl pooling": "CXL"}, near_misses=[],
                         stage_counts={"mined": 100, "eligible": 90,
                                       "novelty-checked": 20, "survivors": 15,
                                       "dossiers": 1})
    assert "# Discovery Agenda — memory-innovation" in text
    assert "2026-07-13-1430" in text
    assert "CXL × PIM" in text
    assert "open gap (2" in text
    assert "T2" in text and "Commodity CXL hardware suffices." in text
    assert "discovery-cxl--pim-2026-07-13-1430.md" in text
    assert "cxl pooling" in text and "→ CXL" in text          # merge map appendix
    assert "mined: 100" in text and "dossiers: 1" in text     # stage counts


def test_agenda_escapes_markdown_pipes_in_table():
    bad = _item(candidate=_cand(a="A|B"))
    text = render_agenda(_meta(), [bad], [], [], {}, [], {})
    assert "A\\|B" in text


def test_agenda_renders_pursuing_and_suppressed():
    text = render_agenda(
        _meta(), [_item()],
        pursuing=[{"key": "DDR6 || MRAM", "first_recommended": "2026-07-01",
                   "last_recommended": "2026-07-10", "best_score": 0.6}],
        suppressed=[("A || B", "rejected")], merge_map={}, near_misses=[],
        stage_counts={})
    assert "DDR6 || MRAM" in text
    assert "A || B" in text and "rejected" in text


def test_agenda_with_no_items_says_so():
    text = render_agenda(_meta(), [], [], [], {}, [], {})
    assert "No opportunities survived filtering" in text


def test_agenda_lists_near_misses_and_degradations():
    meta = _meta()
    meta["degraded"] = ["Semantic Scholar unavailable — novelty unverified"]
    text = render_agenda(meta, [_item()], [], [], {}, near_misses=[_cand("HBM4", "DDR6")],
                         stage_counts={})
    assert "HBM4 × DDR6" in text
    assert "Semantic Scholar unavailable" in text

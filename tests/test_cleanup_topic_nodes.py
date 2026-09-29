import cleanup_topic_nodes as ct


def test_parenthetical_variants_collapse():
    names = [
        "Emerging Memory",
        "Emerging Memory (ReRAM)",
        "Emerging Memory (SOT-MRAM)",
        "Emerging Memory (PCM)",
    ]
    plan = ct.plan_merges(names)
    assert len(plan) == 1
    target, sources = plan[0]
    assert target == "Emerging Memory"  # paren-free, shortest
    assert set(sources) == {
        "Emerging Memory (ReRAM)",
        "Emerging Memory (SOT-MRAM)",
        "Emerging Memory (PCM)",
    }


def test_case_and_punct_variants_collapse():
    names = [
        "System Level Memory Innovation",
        "System level memory innovation",
        "System-level memory innovation",
    ]
    plan = ct.plan_merges(names)
    assert len(plan) == 1
    target, sources = plan[0]
    assert len(sources) == 2
    assert target not in sources


def test_distinct_labels_preserved():
    # Different base labels must NOT merge.
    names = ["HBM", "HBM3", "HBM3E", "HBM4", "HBM (HBM3E)"]
    plan = ct.plan_merges(names)
    # Only "HBM" and "HBM (HBM3E)" share a key -> one group; the spec'd HBM3/E/4 stay.
    assert len(plan) == 1
    target, sources = plan[0]
    assert target == "HBM"
    assert sources == ["HBM (HBM3E)"]


def test_singletons_produce_no_merges():
    names = ["HBM3", "DDR5", "CXL"]
    assert ct.plan_merges(names) == []


def test_target_prefers_paren_free_even_when_longer():
    names = ["CXL (memory pooling)", "CXL Fabric Interconnect Architecture", "CXL"]
    # "CXL" and "CXL (memory pooling)" share key 'cxl'; the third is distinct.
    plan = ct.plan_merges(names)
    target, sources = next(p for p in plan if p[0] in ("CXL", "CXL (memory pooling)"))
    assert target == "CXL"
    assert sources == ["CXL (memory pooling)"]


def test_plan_sorted_by_impact():
    names = (
        ["A", "A (x)"]  # 1 dup
        + ["B", "B (x)", "B (y)", "B (z)"]  # 3 dups
    )
    plan = ct.plan_merges(names)
    assert [t for t, _ in plan] == ["B", "A"]  # largest group first

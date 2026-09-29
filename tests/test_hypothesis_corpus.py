from datetime import date

import networkx as nx

from mira.hypothesis.corpus import Corpus, match_anchor_topics, report_date


def make_fixture_graph() -> nx.Graph:
    """Mirror of the real GraphML shape: undirected, entity_type node attr,
    edge type in the `keywords` edge attr, Report nodes named
    '{profile}-{YYYY-MM-DD}'."""
    g = nx.Graph()
    for name, etype in [
        ("test-2026-05-01", "Report"), ("test-2026-05-08", "Report"),
        ("other-2026-05-01", "Report"),
        ("P1", "Paper"), ("P2", "Paper"), ("P3", "Paper"), ("P4", "Paper"),
        ("P5", "Paper"),  # belongs to the 'other' profile only
        ("CXL", "Topic"), ("PIM", "Topic"), ("HBM4", "Topic"), ("DDR6", "Topic"),
        ("Samsung", "Institution"), ("SK Hynix", "Institution"),
        ("Kim", "Author"), ("Lee", "Author"),
    ]:
        g.add_node(name, entity_type=etype)
    def edge(u, v, kw):
        g.add_edge(u, v, keywords=kw)
    # Report membership
    edge("test-2026-05-01", "P1", "selected_in report")
    edge("test-2026-05-01", "P2", "selected_in report")
    edge("test-2026-05-08", "P3", "selected_in report")
    edge("test-2026-05-08", "P4", "selected_in report")
    edge("other-2026-05-01", "P5", "selected_in report")
    # Paper -> Topic
    edge("P1", "CXL", "primary_topic topic")
    edge("P2", "PIM", "primary_topic topic")
    edge("P3", "CXL", "primary_topic topic")
    edge("P3", "HBM4", "also_covers topic")   # P3 connects CXL and HBM4
    edge("P4", "DDR6", "primary_topic topic")
    edge("P5", "PIM", "primary_topic topic")  # other profile; must be excluded
    # Paper -> Author
    edge("P1", "Kim", "authored_by author")
    edge("P2", "Kim", "authored_by author")   # Kim publishes on CXL and PIM
    edge("P4", "Lee", "authored_by author")
    # Institution -> Topic (researches)
    edge("Samsung", "CXL", "researches topic")
    edge("Samsung", "PIM", "researches topic")
    edge("SK Hynix", "CXL", "researches topic")
    # Topic -> Topic
    edge("CXL", "HBM4", "related_to topic")
    edge("PIM", "HBM4", "related_to topic")
    return g


def make_multi_profile_fixture_graph() -> nx.Graph:
    g = make_fixture_graph()
    g.add_node("Photonics", entity_type="Topic")
    # P2 is deliberately selected in both profiles; P5 and Photonics exist
    # only through the second profile.
    g.add_edge("other-2026-05-01", "P2", keywords="selected_in report")
    g.add_edge("P5", "Photonics", keywords="primary_topic topic")
    return g


def test_report_date_parses_trailing_iso_date():
    assert report_date("memory-innovation-2026-05-04") == date(2026, 5, 4)
    assert report_date("no-date-here") is None


def test_corpus_scopes_papers_to_profile():
    corpus = Corpus.build(make_fixture_graph(), "test")
    assert corpus.papers == {"P1", "P2", "P3", "P4"}
    assert "P5" not in corpus.papers
    assert corpus.report_dates == [date(2026, 5, 1), date(2026, 5, 8)]
    assert corpus.paper_dates["P1"] == date(2026, 5, 1)
    assert corpus.paper_dates["P3"] == date(2026, 5, 8)


def test_single_profile_build_preserves_current_papers_topics_and_dates():
    """A collection-aware build must not change the established one-profile view."""
    corpus = Corpus.build(make_fixture_graph(), "test")

    assert corpus.papers == {"P1", "P2", "P3", "P4"}
    assert corpus.paper_dates == {
        "P1": date(2026, 5, 1),
        "P2": date(2026, 5, 1),
        "P3": date(2026, 5, 8),
        "P4": date(2026, 5, 8),
    }
    assert corpus.topic_papers == {
        "CXL": {"P1", "P3"},
        "PIM": {"P2"},
        "HBM4": {"P3"},
        "DDR6": {"P4"},
    }


def test_multi_profile_build_includes_each_profiles_papers_and_topics():
    """Selecting multiple profiles must retain their union, including unique topics."""
    corpus = Corpus.build(make_multi_profile_fixture_graph(), ("test", "other"))

    assert corpus.profile_ids == ("test", "other")
    assert corpus.papers == {"P1", "P2", "P3", "P4", "P5"}
    assert corpus.topic_papers["Photonics"] == {"P5"}


def test_multi_profile_build_records_each_papers_selecting_profiles():
    """A shared paper must retain provenance from every selecting profile."""
    corpus = Corpus.build(make_multi_profile_fixture_graph(), ("test", "other"))

    assert corpus.paper_profiles["P1"] == {"test"}
    assert corpus.paper_profiles["P2"] == {"test", "other"}
    assert corpus.paper_profiles["P5"] == {"other"}


def test_unknown_profile_build_returns_empty_corpus():
    corpus = Corpus.build(make_fixture_graph(), ("unknown",))

    assert corpus.profile_ids == ("unknown",)
    assert corpus.papers == set()
    assert corpus.paper_profiles == {}
    assert corpus.topic_papers == {}


def test_corpus_since_filter_drops_older_reports():
    corpus = Corpus.build(make_fixture_graph(), "test", since=date(2026, 5, 5))
    assert corpus.papers == {"P3", "P4"}


def test_topic_maps():
    corpus = Corpus.build(make_fixture_graph(), "test")
    assert corpus.topic_papers["CXL"] == {"P1", "P3"}
    assert corpus.topic_papers["PIM"] == {"P2"}
    assert corpus.topic_authors["CXL"] == {"Kim"}
    assert corpus.topic_authors["PIM"] == {"Kim"}
    assert corpus.topic_institutions["CXL"] == {"Samsung", "SK Hynix"}
    assert corpus.topic_institutions["PIM"] == {"Samsung"}
    assert corpus.related_topics("CXL") == {"HBM4"}
    assert corpus.related_topics("PIM") == {"HBM4"}


def test_match_anchor_topics_exact_then_substring():
    topics = ["CXL", "PIM", "HBM4", "CXL Memory Pooling"]
    # normalize_topic-style exact match wins (case-insensitive)
    assert match_anchor_topics("cxl", topics) == ["CXL"]
    # substring both ways when no exact match
    assert match_anchor_topics("memory pooling", topics) == ["CXL Memory Pooling"]
    assert match_anchor_topics("quantum", topics) == []


def test_profile_prefix_collision_is_not_absorbed():
    g = make_fixture_graph()
    # Profile 'test-extended' shares the prefix 'test-'; its papers must NOT
    # leak into profile 'test', and vice versa.
    g.add_node("test-extended-2026-05-01", entity_type="Report")
    g.add_node("P9", entity_type="Paper")
    g.add_edge("test-extended-2026-05-01", "P9", keywords="selected_in report")
    assert "P9" not in Corpus.build(g, "test").papers
    assert Corpus.build(g, "test-extended").papers == {"P9"}

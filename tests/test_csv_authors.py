import csv_authors as ca


def test_keeps_clean_matching_institution():
    raw = {"Genghan Zhang": ["Stanford University"]}
    clean = {"Stanford University", "SambaNova Systems"}
    assert ca.filter_author_affiliations(raw, clean) == {"Genghan Zhang": ["Stanford University"]}


def test_drops_messy_non_matching_institution():
    raw = {"A Researcher": ["Department of Chemistry, University of Southern California, LA, CA, USA"]}
    clean = {"University of Southern California"}  # author string doesn't exactly match
    out = ca.filter_author_affiliations(raw, clean)
    assert out == {"A Researcher": []}  # author kept, messy institution dropped


def test_author_with_no_institution_kept_empty():
    raw = {"Solo Author": []}
    assert ca.filter_author_affiliations(raw, {"MIT"}) == {"Solo Author": []}


def test_blank_author_skipped():
    raw = {"  ": ["MIT"], "Real Name": ["MIT"]}
    out = ca.filter_author_affiliations(raw, {"MIT"})
    assert out == {"Real Name": ["MIT"]}


def test_multiple_institutions_partial_match():
    raw = {"Bridge Author": ["MIT", "Some Lab, Building 5, Cambridge, MA"]}
    clean = {"MIT", "Stanford University"}
    assert ca.filter_author_affiliations(raw, clean) == {"Bridge Author": ["MIT"]}


def test_empty_input():
    assert ca.filter_author_affiliations({}, {"MIT"}) == {}
    assert ca.filter_author_affiliations(None, {"MIT"}) == {}

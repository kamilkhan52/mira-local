"""The matcher decides which Author nodes get deleted from the production graph,
so its false-positive behaviour (deleting a real researcher) is what these tests
pin hardest.
"""
from mira.author_validate import is_real_author, name_tokens


def test_name_tokens_folds_accents_and_drops_initials():
    assert name_tokens("José García-López") == {"jose", "garcia", "lopez"}
    assert name_tokens("A. M. Kalashnikova") == {"kalashnikova"}
    assert name_tokens("vivas2") == {"vivas"}
    assert name_tokens("u") == frozenset()
    assert name_tokens("") == frozenset()


# ── real names must survive ───────────────────────────────────────────────────

def test_truncated_surname_matches_full_name():
    # "Chen" is a real node truncated from an arXiv "Wei Chen".
    assert is_real_author("Chen", ["Wei Chen", "Ada Lovelace"])


def test_single_surname_matches():
    assert is_real_author("Indiveri", ["Giacomo Indiveri"])


def test_extra_middle_name_still_matches():
    # arXiv omits the middle name; the real node must not be deleted.
    assert is_real_author("John Michael Smith", ["John Smith"])


def test_accent_and_romanization_differences_match():
    assert is_real_author("Jose Garcia", ["José García"])


def test_glued_fragment_that_is_a_real_person_survives():
    # "allan.pinto" normalizes to {allan, pinto} and matches "Allan Pinto".
    assert is_real_author("allan.pinto", ["Allan Pinto"])


def test_split_compound_surname_matches_joined_arxiv_form():
    # The CSV split "McDanel" -> "Mc Danel"; arXiv keeps it joined. Tokens
    # differ, letters do not, so the real person must survive.
    assert is_real_author("Bradley Mc Danel", ["Bradley McDanel"])
    assert is_real_author("Amy Siobhan Mc Keown-Green", ["Amy Siobhan McKeown-Green"])
    assert is_real_author("Jarrod R. Mc Clean", ["Jarrod R. McClean"])


def test_glued_initial_and_surname_matches():
    # "Ali RButt" is "Ali R. Butt" with the initial glued to the surname.
    assert is_real_author("Ali RButt", ["Ali R. Butt"])


def test_shortened_given_name_matches():
    # "Ben Chaffin" is the real "Benjamin Chaffin".
    assert is_real_author("Ben Chaffin", ["Benjamin Chaffin", "Atiq Bajwa"])
    # ...but a different surname must not be rescued by the nickname rule.
    assert not is_real_author("Ben Chaffin", ["Benjamin Franklin"])


def test_accent_drop_split_matches():
    # "Sébastien Le Beux" lost its accent to a space -> "S bastien Le Beux".
    assert is_real_author("S bastien Le Beux", ["Sébastien Le Beux", "Masoud Rahimi"])


def test_edit_distance_rule_does_not_rescue_short_junk():
    # "Wait" is 4 letters — below the length gate, so no fuzzy rescue.
    assert not is_real_author("Wait", ["Wei", "Walt Chen"])


def test_animal_substitution_still_rejected_after_fuzzy_rules():
    # "elephant"/"eleonora" share a 3-char prefix ("ele") but the surname rule
    # requires it AND they are >1 edit apart, so the corruption stays rejected.
    assert not is_real_author("Elephant Giunchiglia", ["Eleonora Giunchiglia"])


# ── junk must be rejected ─────────────────────────────────────────────────────

def test_non_name_word_is_rejected():
    assert not is_real_author("Drafting", ["Wei Chen", "Giacomo Indiveri"])
    assert not is_real_author("Alphabetical", ["Wei Chen"])


def test_animal_substitution_is_rejected_even_when_surname_appears():
    # The corrupted node shares only the surname; "elephant" is in no arXiv
    # author, so full containment fails and it is rejected — while the real
    # "Eleonora Giunchiglia" (if present as its own node) would be kept.
    assert not is_real_author("Elephant Giunchiglia", ["Eleonora Giunchiglia"])


def test_empty_token_name_is_rejected():
    assert not is_real_author("u", ["Wei Chen"])
    assert not is_real_author("p", ["Giacomo Indiveri"])


def test_no_arxiv_authors_means_no_match():
    assert not is_real_author("Wei Chen", [])


def test_unrelated_real_name_is_rejected():
    # A real-looking node that authored none of the paper's arXiv authors.
    assert not is_real_author("Marie Curie", ["Wei Chen", "Giacomo Indiveri"])

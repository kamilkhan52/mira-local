"""Decide whether an Author node's name is a real person, using arXiv authors
as ground truth.

The upstream CSV extraction produced three kinds of bad Author names that no
pattern can separate from real ones: non-name words ("Drafting", "Alphabetical"),
glued fragments ("allan.pinto", "vivas2"), and animal-substitution corruptions
("Elephant Giunchiglia" for "Eleonora Giunchiglia"). "Chen" and "Indiveri" are
real; "Drafting" is not — and only the paper's actual author list can tell them
apart.

So this module matches a node name against the arXiv author names of the papers
it is attached to. The matcher is deliberately lenient — a real researcher must
never be deleted (over-rejection is the worse error, per graph_ingest), so a node
survives on any plausible overlap.

Spec: docs/superpowers/specs/2026-07-16-optical-graphrag-parity-design.md (author
provenance from arXiv)
"""
from __future__ import annotations

import re
import unicodedata


def _fold(name: str) -> str:
    """Lowercase and strip accents."""
    folded = unicodedata.normalize("NFKD", name or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).lower()


def name_tokens(name: str) -> frozenset[str]:
    """Normalize a name to comparable tokens.

    Lowercase, strip accents, drop anything non-alphabetic to spaces, and keep
    tokens of length >= 2 (single letters are initials — noise for matching).
    "José García-López" -> {jose, garcia, lopez}; "A. M. Kalashnikova" ->
    {kalashnikova}; "vivas2" -> {vivas}.
    """
    tokens = re.split(r"[^a-zA-Z]+", _fold(name))
    return frozenset(t for t in tokens if len(t) >= 2)


def _letters(name: str) -> str:
    """Every letter, in order, nothing else. "Ali R. Butt" and "Ali RButt" both
    become "alirbutt"; "Mc Keown-Green" and "McKeown-Green" both "mckeowngreen"."""
    return re.sub(r"[^a-z]", "", _fold(name))


def _ordered_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z]+", _fold(name)) if len(t) >= 2]


def _within_edit_distance_1(a: str, b: str) -> bool:
    """True if a and b differ by at most one insertion/deletion/substitution.

    Used only on long letters-only forms, to rescue names an accent-drop split
    ("Sébastien" -> "S bastien" -> letters "sbastien" vs "sebastien")."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:  # one deletion from a
        a, b, la, lb = b, a, lb, la
    i = j = edits = 0  # a is shorter; try to align with one insertion in b
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            j += 1
            if edits > 1:
                return False
    return True


def _prefix_compatible(x: str, y: str) -> bool:
    """One token is a >=3-char prefix of the other — a shortened given name
    ("ben" of "benjamin"), not a coincidental one-letter overlap."""
    lo, hi = sorted((x, y), key=len)
    return len(lo) >= 3 and hi.startswith(lo)


def is_real_author(node_name: str, arxiv_author_names: list[str]) -> bool:
    """True if node_name plausibly denotes one of the given arXiv authors.

    A node matches an arXiv author when either name's token set contains the
    other's — so a truncated node ("Chen") matches "Wei Chen", and a node with an
    extra middle name matches an arXiv name that omits it. A node with no usable
    tokens (e.g. "u", "p") matches nothing.

    Requiring full containment (not just a shared token) is what rejects the
    animal-substitutions: "Elephant Giunchiglia" -> {elephant, giunchiglia} is
    contained in no single arXiv author, because none carries "elephant" — even
    though "giunchiglia" appears in the real "Eleonora Giunchiglia".
    """
    node = name_tokens(node_name)
    if not node:
        return False
    node_letters = _letters(node_name)
    node_ord = _ordered_tokens(node_name)
    for cand in arxiv_author_names:
        a = name_tokens(cand)
        if not a:
            continue
        cand_letters = _letters(cand)
        # Token containment handles truncation and extra middle names; letters-
        # only equality handles compound surnames the CSV split or glued
        # ("Mc Danel" vs "McDanel", "Ali RButt" vs "Ali R. Butt").
        if node <= a or a <= node or node_letters == cand_letters:
            return True
        # Accent-drop that split a token ("Sébastien" -> "S bastien"): the
        # letters-only forms are a single edit apart. Length-gated so short junk
        # can't slip through on coincidence.
        if len(node_letters) >= 8 and _within_edit_distance_1(node_letters, cand_letters):
            return True
        # Shared surname + a shortened given name ("Ben" of "Benjamin Chaffin").
        cand_ord = _ordered_tokens(cand)
        if (node_ord and cand_ord and node_ord[-1] == cand_ord[-1]
                and _prefix_compatible(node_ord[0], cand_ord[0])):
            return True
    return False

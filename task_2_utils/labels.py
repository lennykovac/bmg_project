"""Leaf names and thinness classes, in the notation of the BMG papers.

Leaf names
----------
Each colour gets a letter -- the colours in increasing order become ``a``,
``b``, ``c`` ... -- and the leaves of one colour are numbered in increasing
order of their vertex id: the leaves of colour ``a`` are ``a1, a2, a3 ...``.

The names depend only on the *leaf set and its colouring*, never on the
inner structure. The gene tree, its BMG, both least resolved trees and every
BIC-cherry network built from them therefore name the same gene the same
way, even though none of those constructions copies the attribute along.
(A graph on a *subset* of the leaves -- e.g. from :func:`minimize_bmg` --
would be renumbered; pass ``reference=`` to keep the original names.)

Thinness classes
----------------
Two vertices of a BMG are *thin-equivalent*, ``x ~ y``, iff they have the
same colour and the same in- and out-neighbourhoods:
``N+(x) = N+(y)`` and ``N-(x) = N-(y)`` (Geiß et al. 2019). No best match
test can tell them apart. In a sink-free BMG the colour condition is implied,
it is kept so the function is well defined on any coloured digraph.

Only classes with **at least two** members get a name: the Greek letter of
their colour's Latin letter (``a -> alpha``, ``b -> beta`` ...), numbered
``alpha1, alpha2 ...`` when one colour has several such classes, bare
``alpha`` when it has one.
"""

from __future__ import annotations

import re
from typing import Hashable

import networkx as nx

__all__ = [
    "GREEK",
    "color_letters",
    "leaf_labels",
    "label_leaves",
    "thinness_classes",
    "class_names",
    "best_match_graph_of",
]

GREEK = "αβγδεζηθικλμνξοπρστυφχψω"
_LATIN = "abcdefghijklmnopqrstuvwxyz"


def _natural(v) -> tuple:
    return tuple(
        (0, int(tok), "") if tok.isdigit() else (1, 0, tok)
        for tok in re.split(r"(\d+)", str(v))
        if tok
    )


def _leaves(G: nx.DiGraph) -> list:
    """Leaves of a network, or every vertex of a BMG (all vertices coloured)."""
    if all(c is not None for _, c in G.nodes(data="color")):
        return list(G)
    return [v for v in G if G.out_degree(v) == 0]


def _nth(alphabet: str, i: int) -> str:
    # a..z, then aa, ab ... -- only needed beyond 26 colours
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, len(alphabet))
        out = alphabet[r] + out
    return out


def _index(alphabet: str, word: str) -> int:
    """Inverse of :func:`_nth`."""
    i = 0
    for ch in word:
        i = i * len(alphabet) + alphabet.index(ch) + 1
    return i - 1


def color_letters(G: nx.DiGraph) -> dict:
    """``colour value -> letter`` (colours in increasing order -> a, b, c ...)."""
    colours = sorted({G.nodes[v].get("color") for v in _leaves(G)}, key=_natural)
    return {c: _nth(_LATIN, i) for i, c in enumerate(colours)}


def leaf_labels(G: nx.DiGraph, reference: nx.DiGraph | None = None) -> dict:
    """``leaf -> "a1"`` ... (see the module docstring).

    A ``label`` attribute already on the leaf wins. With ``reference``, leaves
    that also occur there take their name from it, so a subgraph keeps the
    names of the instance it was cut from.
    """
    source = reference if reference is not None else G
    letters = color_letters(source)
    by_colour: dict = {}
    for v in _leaves(source):
        by_colour.setdefault(source.nodes[v].get("color"), []).append(v)

    computed: dict = {}
    for c, members in by_colour.items():
        for i, v in enumerate(sorted(members, key=_natural), start=1):
            computed[v] = f"{letters[c]}{i}"

    out = {}
    for v in _leaves(G):
        stored = G.nodes[v].get("label")
        if stored is None and reference is not None and v in reference:
            stored = reference.nodes[v].get("label")
        out[v] = stored if stored is not None else computed.get(v, str(v))
    return out


def label_leaves(G: nx.DiGraph, reference: nx.DiGraph | None = None) -> dict:
    """Write :func:`leaf_labels` into the ``label`` attribute; returns the map."""
    labels = leaf_labels(G, reference)
    nx.set_node_attributes(G, labels, "label")
    return labels


def best_match_graph_of(G: nx.DiGraph) -> nx.DiGraph:
    """``G`` itself if it is already a BMG, else the BMG it explains."""
    if G.number_of_nodes() and all(c is not None for _, c in G.nodes(data="color")):
        return G
    try:
        from utils.fast_bmg import fast_bmg
        return fast_bmg(G)[0]
    except ImportError:  # pragma: no cover - fast_bmg is optional
        from utils.graph_utils import bmg_from_network
        return bmg_from_network(G)


def thinness_classes(G: nx.DiGraph) -> list[frozenset]:
    """All thinness classes of the BMG of ``G`` (singletons included).

    ``G`` may be the BMG itself or any network explaining it (gene tree,
    LRT, BIC-cherry network) -- then its BMG is computed first. Sorted by the
    smallest member id.
    """
    bmg = best_match_graph_of(G)
    buckets: dict = {}
    for v in bmg:
        key = (
            bmg.nodes[v].get("color"),
            frozenset(bmg.successors(v)),
            frozenset(bmg.predecessors(v)),
        )
        buckets.setdefault(key, set()).add(v)
    classes = [frozenset(s) for s in buckets.values()]
    return sorted(classes, key=lambda s: min(_natural(v) for v in s))


def class_names(
    G: nx.DiGraph,
    labels: dict | None = None,
    classes: list[frozenset] | None = None,
) -> dict:
    """``leaf -> "α1"`` for every leaf in a thinness class of size >= 2.

    Leaves in singleton classes are absent from the result. Classes of one
    colour are numbered in the order of their smallest leaf *name*; the
    number is dropped when the colour has only one named class.
    """
    labels = labels if labels is not None else leaf_labels(G)
    classes = classes if classes is not None else thinness_classes(G)
    letters = color_letters(G)

    def greek(cls) -> str:
        # the Greek letter follows the Latin letter of the members' names
        # (a -> α), so names taken over from a reference stay consistent
        name = str(labels.get(min(cls, key=_natural), ""))
        m = re.match(r"[a-z]+", name)
        latin = m.group(0) if m else letters[G.nodes[next(iter(cls))].get("color")]
        return _nth(GREEK, _index(_LATIN, latin))

    per_colour: dict = {}
    for cls in classes:
        if len(cls) < 2:
            continue
        colour = G.nodes[next(iter(cls))].get("color")
        per_colour.setdefault(colour, []).append(cls)

    names = {}
    for colour, group in per_colour.items():
        group.sort(key=lambda s: min(_natural(labels.get(v, v)) for v in s))
        for i, cls in enumerate(group, start=1):
            name = greek(cls) + (str(i) if len(group) > 1 else "")
            for v in cls:
                names[v] = name
    return names

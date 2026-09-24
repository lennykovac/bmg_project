"""Structure-aware plots of gene trees, BMGs, least resolved trees and
BIC-cherry networks (interactive HTML via pyvis, static node positions).

Layouts
-------
``tree`` / ``bic_cherry`` / ``network``  -- layered, top-down
    * the root sits on the top row, **all** leaves on the bottom row;
    * every inner vertex sits on the row of its depth, i.e. the length of the
      longest root-to-vertex path. For a tree that is the ordinary depth; for
      a BIC-cherry expansion it gives ``R`` -> ``p:`` -> ``q:`` -> leaves;
    * positions are computed here and the nodes are ``fixed``: no physics, no
      dragging (pan and zoom still work).

``bmg``  -- colour groups
    * reciprocal best matches ``x <-> y`` are drawn as ONE edge with a head
      on each end (gray); one-way best matches ``x -> y`` keep a single head
      (red), so the asymmetric part of the BMG stands out;
    * two colours: one column per colour, left and right, arcs through the
      middle;
    * more colours: the groups sit around a circle, each group a short row
      tangent to it, so every arc crosses the interior.

Determinism
-----------
The layout depends only on the *cluster structure* and the leaf labels, never
on node ids or insertion order. Children are ordered by the smallest leaf
label below them, and inner vertices of trees get canonical display labels
(``v0`` = root, then preorder). Two trees with ``same_phylogeny(T1, T2)``
therefore produce the same picture -- in particular ``LRT(T)`` and
``LRT(G(T))``, whose inner ids (``i0``, ``i1`` ...) come out of tralda in
different orders. The original id stays visible in the tooltip.

Leaf names and thinness classes
-------------------------------
Leaves are shown as ``a1, a2, b1 ...`` (one letter per colour, see
:mod:`task_2_utils.labels`); the original id is in the tooltip. Individual leaves
are always round. A subtitle lists every thinness class with >= 2 leaves --
``α`` for colour ``a``, ``α1``/``α2`` when colour ``a`` has several -- and
its members.

* BMG: the members stay individual round leaves, placed next to each other,
  and each class is circled by a dashed outline in its colour that carries
  the class name.
* least resolved tree: each class is drawn as ONE vertex of its own geometry
  (square, diamond, triangle ...) labelled with the class name, in place of
  its members (the thin quotient, :func:`collapse_thin_classes`).
* gene tree (and networks): the members stay individual round leaves where
  the tree puts them, with the class name written under their names.

``plot_graph(..., collapse_classes=..., enclose_classes=...)`` overrides that.

Colours
-------
Leaves: fixed palette indexed by the colour value (ints directly, anything
else via a stable CRC32 -- *not* ``hash()``, which is salted per process), so
the same species gets the same colour in every plot of every run. Inner
vertices: grayscale by role -- root ``#404040``, cherry ``p:`` ``#808080``,
expansion ``q:`` ``#B0B0B0``, other inner ``#9A9A9A``.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from typing import Hashable, Mapping, Optional

import networkx as nx
from pyvis.network import Network

from task_2_utils.labels import class_names, leaf_labels

__all__ = [
    "PALETTE",
    "INNER_GRAYS",
    "detect_graph_type",
    "node_color",
    "layout",
    "display_labels",
    "bmg_edges",
    "leaf_rank",
    "collapse_thin_classes",
    "CLASS_SHAPES",
    "plot_graph",
    "plot_comparison",
]

PALETTE = (
    "#E4572E",  # red
    "#17BEBB",  # teal
    "#3A6EA5",  # blue
    "#F2A541",  # orange
    "#76B041",  # green
    "#8E5572",  # plum
    "#FFC914",  # yellow
    "#2E282A",  # near-black
    "#C3423F",  # brick
    "#5BC0EB",  # sky
)

INNER_GRAYS = {
    "root": "#404040",
    "cherry": "#808080",
    "expansion": "#B0B0B0",
    "inner": "#9A9A9A",
}

LEAF_GAP = 80      # minimal horizontal distance between neighbouring leaves (px)
INNER_GAP = 60     # minimal horizontal distance inside an inner row (px)
LEVEL_GAP = 110    # minimal vertical distance between rows (px)
ASPECT = 0.45      # height / width the layered layout aims for
BMG_RADIUS = 260   # half-distance between colour groups in a BMG
BMG_ROW_GAP = 70   # distance between vertices inside one colour group

#: node shapes of the thinness classes (singletons stay ``dot``)
CLASS_SHAPES = ("square", "triangle", "diamond", "star", "triangleDown", "hexagon")


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def _is_cherry(v) -> bool:
    return isinstance(v, str) and v.startswith("p:")


def _is_expansion(v) -> bool:
    return isinstance(v, str) and v.startswith("q:")


def detect_graph_type(G: nx.DiGraph) -> str:
    """``"bmg"``, ``"bic_cherry"``, ``"tree"`` or ``"network"``.

    * bmg        -- every vertex carries a colour (no inner vertices at all);
    * bic_cherry -- contains ``p:``/``q:`` vertices (tree-shaped or not, so
                    edited networks from the search are recognised as well);
    * tree       -- single root, in-degree <= 1;
    * network    -- anything else with inner vertices.
    """
    if G.number_of_nodes() and all(c is not None for _, c in G.nodes(data="color")):
        return "bmg"
    if any(_is_cherry(v) or _is_expansion(v) for v in G):
        return "bic_cherry"
    roots = [v for v in G if G.in_degree(v) == 0]
    if (
        len(roots) == 1
        and all(G.in_degree(v) <= 1 for v in G)
        and nx.is_directed_acyclic_graph(G)
    ):
        return "tree"
    return "network"


def _role(G: nx.DiGraph, v) -> str:
    if G.out_degree(v) == 0:
        return "leaf"
    if G.in_degree(v) == 0:
        return "root"
    if _is_cherry(v):
        return "cherry"
    if _is_expansion(v):
        return "expansion"
    return "inner"


def node_color(G: nx.DiGraph, v, color_map: Mapping | None = None) -> str:
    """HTML colour of ``v`` (see module docstring)."""
    c = G.nodes[v].get("color")
    if c is not None:
        if color_map is not None and c in color_map:
            return color_map[c]
        if isinstance(c, bool):
            c = int(c)
        if isinstance(c, int):
            return PALETTE[c % len(PALETTE)]
        return PALETTE[zlib.crc32(str(c).encode()) % len(PALETTE)]
    return INNER_GRAYS.get(_role(G, v), INNER_GRAYS["inner"])


# ---------------------------------------------------------------------------
# ordering helpers
# ---------------------------------------------------------------------------

def _natural(v) -> tuple:
    """Sort key that orders ``2 < 10`` and never compares int with str."""
    return tuple(
        (0, int(tok), "") if tok.isdigit() else (1, 0, tok)
        for tok in re.split(r"(\d+)", str(v))
        if tok
    )


def _clusters(G: nx.DiGraph, order: list) -> dict:
    """``v -> frozenset`` of leaves below ``v`` (for a DAG in topological order)."""
    below: dict = {}
    for v in reversed(order):
        if G.out_degree(v) == 0:
            below[v] = frozenset([v])
        else:
            below[v] = frozenset().union(*(below[c] for c in G.successors(v)))
    return below


def _leaf_key(v, rank: Mapping | None) -> tuple:
    """Position of a leaf in the global leaf order (``rank``), else its id."""
    if rank is not None and v in rank:
        return (0, rank[v])
    return (1, _natural(v))


def _cluster_key(cluster: frozenset, rank: Mapping | None = None) -> tuple:
    leaves = sorted(_leaf_key(x, rank) for x in cluster)
    return (leaves[0] if leaves else (), len(leaves), tuple(leaves))


def _sorted_children(G, v, cluster, rank: Mapping | None = None) -> list:
    return sorted(G.successors(v), key=lambda c: (_cluster_key(cluster[c], rank), _natural(c)))


def leaf_rank(names: Mapping, classes: Mapping) -> dict:
    """Global leaf order: by name, except that the members of a thinness
    class are pulled together behind their first member. ``names`` from
    :func:`task_2_utils.labels.leaf_labels`, ``classes`` from
    :func:`task_2_utils.labels.class_names`."""
    first: dict = {}
    for v, cls in classes.items():
        key = _natural(names[v])
        first[cls] = min(first.get(cls, key), key)
    order = sorted(
        names,
        key=lambda v: (first.get(classes.get(v), _natural(names[v])), _natural(names[v])),
    )
    return {v: i for i, v in enumerate(order)}


# ---------------------------------------------------------------------------
# layered layout (trees and networks)
# ---------------------------------------------------------------------------

def _levels(G: nx.DiGraph, order: list) -> dict:
    """Longest root-to-vertex path; every leaf is pushed to the bottom row."""
    level = {v: 0 for v in G}
    for v in order:
        for c in G.successors(v):
            level[c] = max(level[c], level[v] + 1)
    bottom = max(level.values(), default=0)
    for v in G:
        if G.out_degree(v) == 0:
            level[v] = bottom
    return level


def _dfs_leaf_order(G, roots, cluster, rank=None) -> list:
    seen, leaves = set(), []

    def visit(v):
        if v in seen:
            return
        seen.add(v)
        if G.out_degree(v) == 0:
            leaves.append(v)
        for c in _sorted_children(G, v, cluster, rank):
            visit(c)

    for r in sorted(roots, key=lambda r: (_cluster_key(cluster[r], rank), _natural(r))):
        visit(r)
    return leaves


def _spread(nodes, x, gap):
    """Push apart vertices of one row that are closer than ``gap`` (keeps the mean)."""
    if len(nodes) < 2:
        return
    row = sorted(nodes, key=lambda v: (x[v], _natural(v)))
    wanted = [x[v] for v in row]
    placed = [wanted[0]]
    for w in wanted[1:]:
        placed.append(max(w, placed[-1] + gap))
    shift = (sum(wanted) - sum(placed)) / len(placed)
    for v, p in zip(row, placed):
        x[v] = p + shift


def _inner_x(G, order, level, leaf_x, tree_like, gap=INNER_GAP) -> dict:
    x = dict(leaf_x)
    for v in reversed(order):
        if G.out_degree(v) == 0:
            continue
        xs = [x[c] for c in G.successors(v)]
        # trees: centre over the outermost children (classic cladogram);
        # networks: barycentre, which keeps cherries near both of their leaves
        x[v] = (min(xs) + max(xs)) / 2 if tree_like else sum(xs) / len(xs)
    rows: dict = {}
    for v in G:
        if G.out_degree(v) > 0:
            rows.setdefault(level[v], []).append(v)
    for lvl in sorted(rows, reverse=True):
        _spread(rows[lvl], x, gap)
    return x


def _crossings(G, x, level) -> int:
    segs = [((x[u], level[u]), (x[v], level[v])) for u, v in G.edges]

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    count = 0
    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 1, len(segs)):
            c, d = segs[j]
            if len({a, b, c, d}) < 4:
                continue  # shared endpoint
            d1, d2 = ccw(a, b, c), ccw(a, b, d)
            d3, d4 = ccw(c, d, a), ccw(c, d, b)
            if d1 * d2 < 0 and d3 * d4 < 0:
                count += 1
    return count


def _layered_layout(G: nx.DiGraph, tree_like: bool, sweeps: int = 6, rank=None) -> dict:
    order = list(nx.topological_sort(G))
    roots = [v for v in G if G.in_degree(v) == 0]
    cluster = _clusters(G, order)
    level = _levels(G, order)

    leaves = _dfs_leaf_order(G, roots, cluster, rank)

    # A BIC-cherry network has ~|L|^2/2 cherries on one row: widen the leaf
    # row to match, so the leaves do not bunch up under a very long row.
    rows: dict = {}
    for v in G:
        rows[level[v]] = rows.get(level[v], 0) + 1
    widest_inner = max((n for l, n in rows.items() if l != max(rows)), default=1)
    width = max(len(leaves) * LEAF_GAP, widest_inner * INNER_GAP)
    leaf_gap = width / max(1, len(leaves))
    place = lambda seq: {v: i * leaf_gap for i, v in enumerate(seq)}  # noqa: E731
    x = _inner_x(G, order, level, place(leaves), tree_like)

    if not tree_like:
        # barycentre sweeps on the leaf order; keep the order with fewest
        # crossings. Deterministic: ties are broken by the current position.
        best = (_crossings(G, x, level), leaves, x)
        for _ in range(sweeps):
            pos = {v: i for i, v in enumerate(leaves)}
            bary = {
                v: sum(x[p] for p in G.predecessors(v)) / max(1, G.in_degree(v))
                for v in leaves
            }
            leaves = sorted(leaves, key=lambda v: (bary[v], pos[v]))
            x = _inner_x(G, order, level, place(leaves), tree_like)
            score = _crossings(G, x, level)
            if score < best[0]:
                best = (score, leaves, x)
        x = best[2]

    # centre horizontally
    mid = (min(x.values()) + max(x.values())) / 2 if x else 0
    span = (max(x.values()) - min(x.values())) if x else 0
    depth = max(level.values(), default=0)
    level_gap = max(LEVEL_GAP, ASPECT * span / depth) if depth else LEVEL_GAP
    return {v: (x[v] - mid, level[v] * level_gap) for v in G}


# ---------------------------------------------------------------------------
# BMG layout
# ---------------------------------------------------------------------------

def _offsets(members: list, classes: Mapping) -> list[float]:
    """Positions along a colour group, centred on 0. A thinness class sits as
    one tight block; blocks are separated by an extra half gap."""
    steps, t = [], 0.0
    for i, v in enumerate(members):
        if i:
            same = v in classes and classes.get(v) == classes.get(members[i - 1])
            t += BMG_ROW_GAP * (0.9 if same else 1.6 if (v in classes or members[i - 1] in classes) else 1.0)
        steps.append(t)
    mid = (steps[0] + steps[-1]) / 2 if steps else 0
    return [x - mid for x in steps]


def _bmg_layout(G: nx.DiGraph, rank: Mapping | None = None, classes: Mapping | None = None) -> dict:
    classes = classes or {}
    groups: dict = {}
    for v, c in G.nodes(data="color"):
        groups.setdefault(c, []).append(v)
    colours = sorted(groups, key=_natural)
    for c in colours:
        groups[c].sort(key=lambda v: _leaf_key(v, rank))

    pos = {}
    if len(colours) <= 2:
        for side, c in enumerate(colours):
            xs = -BMG_RADIUS if side == 0 else BMG_RADIUS
            for v, y in zip(groups[c], _offsets(groups[c], classes)):
                pos[v] = (xs, y)
        return pos

    k = len(colours)
    longest = max(abs(o) for g in groups.values() for o in _offsets(g, classes) or [0]) * 2
    radius = max(BMG_RADIUS, (longest + BMG_ROW_GAP) * k / (2 * math.pi))
    for i, c in enumerate(colours):
        phi = -math.pi / 2 + 2 * math.pi * i / k  # first group on top
        cx, cy = radius * math.cos(phi), radius * math.sin(phi)
        tx, ty = -math.sin(phi), math.cos(phi)    # tangent direction
        for v, t in zip(groups[c], _offsets(groups[c], classes)):
            pos[v] = (cx + t * tx, cy + t * ty)
    return pos


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _leaf_context(G: nx.DiGraph, reference=None, with_classes: bool = True):
    """``(names, classes, rank)`` of the leaves of ``G`` (see :mod:`task_2_utils.labels`)."""
    names = leaf_labels(G, reference)
    classes = class_names(G, names) if with_classes else {}
    return names, classes, leaf_rank(names, classes)


def layout(
    G: nx.DiGraph,
    graph_type: str | None = None,
    rank: Mapping | None = None,
    classes: Mapping | None = None,
) -> dict:
    """``v -> (x, y)`` in screen coordinates (y grows downwards).

    ``rank``/``classes`` default to the leaf names and thinness classes of
    ``G`` (:func:`_leaf_context`): leaves are ordered by name, the members of
    a thinness class kept next to each other.
    """
    graph_type = graph_type or detect_graph_type(G)
    if rank is None:
        _, auto_classes, rank = _leaf_context(G)
        classes = auto_classes if classes is None else classes
    if graph_type == "bmg":
        return _bmg_layout(G, rank, classes)
    if not nx.is_directed_acyclic_graph(G):
        # not a BMG and not a DAG: nothing structural to respect
        return {v: (400 * x, 400 * y) for v, (x, y) in nx.circular_layout(G).items()}
    return _layered_layout(G, tree_like=(graph_type == "tree"), rank=rank)


def display_labels(
    G: nx.DiGraph,
    graph_type: str | None = None,
    canonical: bool = True,
    rank: Mapping | None = None,
    names: Mapping | None = None,
) -> dict:
    """Leaves show their name (``a1``, ``b2`` ... from :mod:`task_2_utils.labels`).
    Inner vertices of *trees* get canonical names ``v0`` (root), ``v1`` ... in
    preorder over the canonical child order, so equal phylogenies get equal
    labels. Networks keep their ids (``R``, ``p:x|y``, ``q:x|z`` carry
    meaning)."""
    graph_type = graph_type or detect_graph_type(G)
    if names is None or rank is None:
        auto_names, _, auto_rank = _leaf_context(G)
        names = auto_names if names is None else names
        rank = auto_rank if rank is None else rank
    labels = {v: str(names.get(v, v)) for v in G}
    if not canonical or graph_type != "tree":
        return labels
    order = list(nx.topological_sort(G))
    cluster = _clusters(G, order)
    counter = 0

    def visit(v):
        nonlocal counter
        if G.out_degree(v) > 0:
            labels[v] = f"v{counter}"
            counter += 1
        for c in _sorted_children(G, v, cluster, rank):
            visit(c)

    for r in (v for v in G if G.in_degree(v) == 0):
        visit(r)
    return labels


def bmg_edges(G: nx.DiGraph) -> list[tuple]:
    """``(u, v, reciprocal)`` -- each reciprocal pair ``u <-> v`` once, every
    one-way arc ``u -> v`` as it is. Deterministic order."""
    out = []
    for u, v in sorted(G.edges, key=lambda e: (_natural(e[0]), _natural(e[1]))):
        if G.has_edge(v, u):
            if _natural(u) < _natural(v):
                out.append((u, v, True))
        else:
            out.append((u, v, False))
    return out


def _class_legend(classes: Mapping, names: Mapping) -> str:
    """``"α = {a1, a3} · β1 = {b1, b2}"`` -- only classes with >= 2 leaves."""
    members: dict = {}
    for v, c in classes.items():
        members.setdefault(c, []).append(str(names.get(v, v)))
    return " \u00b7 ".join(
        f"{c} = {{{', '.join(sorted(m, key=_natural))}}}"
        for c, m in sorted(members.items(), key=lambda kv: _natural(kv[0]))
    )


#: draws the outline of every thinness class under the graph (vis-network's
#: ``beforeDrawing`` hook -- pyvis keeps the network in the global ``network``)
_HULL_JS = """<script>
(function () {
  const HULLS = __HULLS__;
  const R = 34;                       // half-width of the outline around a vertex
  function rgba(hex, a) {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function capsule(ctx, cx, cy, ux, uy, half, r) {
    // rounded rectangle along the unit direction (ux, uy)
    const nx = -uy, ny = ux, a = [cx - ux * half, cy - uy * half], b = [cx + ux * half, cy + uy * half];
    const ang = Math.atan2(uy, ux);
    ctx.beginPath();
    ctx.moveTo(a[0] + nx * r, a[1] + ny * r);
    ctx.lineTo(b[0] + nx * r, b[1] + ny * r);
    ctx.arc(b[0], b[1], r, ang + Math.PI / 2, ang - Math.PI / 2, true);
    ctx.lineTo(a[0] - nx * r, a[1] - ny * r);
    ctx.arc(a[0], a[1], r, ang - Math.PI / 2, ang + Math.PI / 2, true);
    ctx.closePath();
  }
  function draw(ctx) {
    for (const h of HULLS) {
      const p = network.getPositions(h.ids), pts = h.ids.map(id => p[id]).filter(Boolean);
      if (pts.length < 2) continue;
      const first = pts[0], last = pts[pts.length - 1];
      let ux = last.x - first.x, uy = last.y - first.y;
      const len = Math.hypot(ux, uy) || 1; ux /= len; uy /= len;
      const cx = (first.x + last.x) / 2, cy = (first.y + last.y) / 2 + 8;  // labels sit below the dots
      capsule(ctx, cx, cy, ux, uy, len / 2, R);
      ctx.fillStyle = rgba(h.color, 0.10); ctx.fill();
      ctx.setLineDash([7, 5]); ctx.lineWidth = 2; ctx.strokeStyle = rgba(h.color, 0.9); ctx.stroke();
      ctx.setLineDash([]);
      // class name on the outer side (away from the middle of the drawing)
      let nx = -uy, ny = ux;
      if (nx * cx + ny * cy < 0) { nx = -nx; ny = -ny; }
      if (Math.hypot(cx, cy) < 1) { nx = 0; ny = -1; }
      ctx.font = "bold 17px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = rgba(h.color, 1);
      ctx.fillText(h.name, cx + nx * (R + 16), cy + ny * (R + 16));
    }
  }
  (function hook() {
    if (typeof network === "undefined" || !network) return setTimeout(hook, 50);
    network.on("beforeDrawing", draw);
    network.redraw();
  })();
})();
</script>"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CLASS_PREFIX = "class:"


def collapse_thin_classes(
    G: nx.DiGraph,
    classes: Mapping,
    graph_type: str | None = None,
) -> tuple[nx.DiGraph, dict]:
    """Replace every named thinness class by ONE vertex ``"class:<name>"``.

    ``classes`` is ``leaf -> class name`` (:func:`task_2_utils.labels.class_names`).
    Returns the quotient graph and ``class vertex -> [member leaves]``.

    * BMG: thin-equivalent leaves have identical in- and out-neighbourhoods,
      so the quotient ``G/~`` is well defined -- ``A -> B`` iff ``a -> b``
      for one (hence every) pair of members.
    * tree: the members have to be siblings -- which they are in a least
      resolved tree -- and are replaced by one leaf under their parent. A
      class whose members sit under different parents is left as it is.
    """
    graph_type = graph_type or detect_graph_type(G)
    groups: dict = {}
    for v, name in classes.items():
        if v in G:
            groups.setdefault(name, []).append(v)

    H = G.copy()
    members: dict = {}
    for name, vs in sorted(groups.items(), key=lambda kv: _natural(kv[0])):
        if len(vs) < 2:
            continue
        if graph_type != "bmg" and len({frozenset(H.predecessors(v)) for v in vs}) != 1:
            continue
        node = CLASS_PREFIX + name
        inside = set(vs)
        H.add_node(node, color=G.nodes[vs[0]].get("color"), thin_class=name)
        for v in vs:
            for p_ in list(H.predecessors(v)):
                if p_ not in inside:
                    H.add_edge(p_, node)
            for c in list(H.successors(v)):
                if c not in inside:
                    H.add_edge(node, c)
        H.remove_nodes_from(vs)
        members[node] = sorted(vs, key=_natural)
    return H, members


def _is_least_resolved(T: nx.DiGraph) -> bool:
    """Is the tree ``T`` its own least resolved tree?"""
    try:
        from utils.graph_utils import same_phylogeny
        from task_2_utils.labels import best_match_graph_of
        from utils.lrt import lrt_from_bmg
    except ImportError:  # pragma: no cover
        return False
    lrt = lrt_from_bmg(best_match_graph_of(T))
    return lrt is not None and same_phylogeny(T, lrt)


def plot_graph(
    G: nx.DiGraph,
    title: Optional[str] = None,
    output_file: str = "graph_plot.html",
    width: str = "100%",
    height: str = "750px",
    graph_type: str | None = None,
    canonical_labels: bool = True,
    color_map: Mapping | None = None,
    show_inner_labels: bool = True,
    show_classes: bool = True,
    collapse_classes: bool | None = None,
    enclose_classes: bool | None = None,
    reference: nx.DiGraph | None = None,
) -> str:
    """Write an HTML plot of ``G`` and return the file name.

    ``graph_type`` overrides the detection; ``canonical_labels=False`` shows
    the raw ids of tree vertices; ``color_map`` maps colour values to HTML
    colours and overrides the palette for those values; ``reference`` is the
    graph to take leaf names from when ``G`` has only some of its leaves.

    Thinness classes (``show_classes``; only classes with >= 2 leaves)
    ------------------------------------------------------------------
    Individual leaves are always round. A subtitle lists every class and its
    members, e.g. ``α = {a1, a2, a3} · β1 = {b1, b4}``.

    ``collapse_classes=True``   each class is drawn as ONE vertex of its own
                                geometry (square, diamond, ...), labelled with
                                the class name, in place of its members
                                (:func:`collapse_thin_classes`).
                                Default: only for a least resolved tree.
    ``enclose_classes=True``    members stay individual round leaves, placed
                                next to each other, and every class is
                                circled by an outline in its colour carrying
                                the class name. Default: for a BMG.
    otherwise                   members stay where they are, with the class
                                name written under each one's name (gene
                                tree, networks).
    """
    graph_type = graph_type or detect_graph_type(G)
    names, classes, rank = _leaf_context(G, reference, with_classes=show_classes)
    legend = _class_legend(classes, names)

    if title is None:
        title = f"{graph_type} ({G.number_of_nodes()} vertices, {G.number_of_edges()} arcs"
        if graph_type == "bmg":
            pairs = sum(both for *_, both in bmg_edges(G))
            title += f", {pairs} reciprocal pairs"
        title += ")"

    if collapse_classes is None:
        collapse_classes = bool(classes) and graph_type == "tree" and _is_least_resolved(G)
    if enclose_classes is None:
        enclose_classes = bool(classes) and graph_type == "bmg" and not collapse_classes
    members: dict = {}
    if collapse_classes and classes:
        G, members = collapse_thin_classes(G, classes, graph_type)
        for node, vs in members.items():
            names[node] = G.nodes[node]["thin_class"]
            rank[node] = min(rank[v] for v in vs)
        # members that were drawn as one vertex need no extra annotation
        classes = {v: c for v, c in classes.items() if v in G}

    shape_of = {c: CLASS_SHAPES[i % len(CLASS_SHAPES)]
                for i, c in enumerate(sorted({G.nodes[n]["thin_class"] for n in members}, key=_natural))}
    pos = layout(G, graph_type, rank=rank, classes=classes)
    labels = display_labels(G, graph_type, canonical=canonical_labels, rank=rank, names=names)

    order = list(nx.topological_sort(G)) if nx.is_directed_acyclic_graph(G) else list(G)
    cluster = _clusters(G, order) if graph_type != "bmg" and nx.is_directed_acyclic_graph(G) else {}

    def shown(v) -> str:
        if v in members:
            return "{" + ", ".join(str(names.get(m, m)) for m in members[v]) + "}"
        return str(names.get(v, v))

    nt = Network(height=height, width=width, directed=True, cdn_resources="in_line")
    pos_by_id = {str(v): pos[v] for v in G}

    # add in a canonical order so the HTML itself is reproducible
    for v in sorted(G, key=lambda v: (pos[v][1], pos[v][0], _natural(v))):
        role = "leaf" if graph_type == "bmg" else _role(G, v)
        is_class = v in members
        tip = [f"thinness class {names[v]}" if is_class else f"id: {v}", f"role: {role}"]
        if is_class:
            tip.append("members: " + shown(v) + "  (ids " + ", ".join(map(str, members[v])) + ")")
        if G.nodes[v].get("color") is not None:
            tip.append(f"color: {G.nodes[v]['color']}")
        if v in classes:
            tip.append(f"thinness class: {classes[v]}")
        if v in cluster and role != "leaf":
            leaves_below = []
            for leaf in cluster[v]:
                leaves_below.extend(members.get(leaf, [leaf]))
            tip.append("cluster: {" + ", ".join(sorted((str(names.get(x, x)) for x in leaves_below), key=_natural)) + "}")
        is_leaf = role == "leaf"
        x, y = pos[v]
        text = labels[v] if (is_leaf or show_inner_labels) else " "
        if v in classes and not enclose_classes:
            text = f"{text}\n{classes[v]}"
        nt.add_node(
            str(v),
            label=text,
            title="\n".join(tip),
            color=node_color(G, v, color_map),
            x=x,
            y=y,
            fixed=True,
            physics=False,
            size=(19 + 2 * min(len(members[v]), 6)) if is_class else (16 if is_leaf else 9),
            shape=shape_of[names[v]] if is_class else "dot",
            font={"size": 15 if is_class else 14 if is_leaf else 11, "color": "#222222"},
            borderWidth=2 if is_class else 1,
        )

    head = {"enabled": True, "scaleFactor": 0.5}
    if graph_type == "bmg":
        # reciprocal best matches x <-> y: ONE edge with a head on each end
        for u, v, both in bmg_edges(G):
            nt.add_edge(
                str(u), str(v),
                color="#8A8A8A" if both else "#C0504D",
                width=1.2,
                arrows={"to": head, "from": head} if both else {"to": head},
                title=(f"{shown(u)} \u2194 {shown(v)}  (reciprocal)" if both
                       else f"{shown(u)} \u2192 {shown(v)}"),
            )
    else:
        for u, v in sorted(G.edges, key=lambda e: (_natural(e[0]), _natural(e[1]))):
            nt.add_edge(str(u), str(v), color="#8A8A8A", width=1.2)

    # no parallel arcs are drawn any more, so every edge can be straight
    smooth = {"enabled": False}
    nt.set_options(json.dumps({
        "physics": {"enabled": False},
        "layout": {"hierarchical": {"enabled": False}},
        "interaction": {
            "dragNodes": False,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "tooltipDelay": 100,
        },
        "edges": {"arrows": {"to": head}, "smooth": smooth},
        "nodes": {"fixed": {"x": True, "y": True}},
    }))

    # pyvis 0.3.2 prints ``heading`` twice -- inject the title ourselves
    html = nt.generate_html(name=output_file)
    header = f'<h3 style="font-family:sans-serif;text-align:center;margin:8px 0">{_escape(title)}</h3>'
    if legend:
        header += ('<p style="font-family:sans-serif;text-align:center;margin:0 0 8px;color:#444">'
                   f"thinness classes: {_escape(legend)}</p>")
    html = html.replace("<body>", "<body>\n" + header, 1)
    if enclose_classes and classes:
        hulls: dict = {}
        for v, c in classes.items():
            if v in G:
                hulls.setdefault(c, {"name": c, "color": node_color(G, v, color_map), "ids": []})["ids"].append(str(v))
        for h in hulls.values():
            h["ids"].sort(key=lambda i: pos_by_id[i][1] * 1e6 + pos_by_id[i][0])
        script = _HULL_JS.replace("__HULLS__", json.dumps(sorted(hulls.values(), key=lambda h: _natural(h["name"]))))
        html = html.replace("</body>", script + "\n</body>", 1)
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"✓ {graph_type} plot saved to {output_file}")
    return output_file


def plot_comparison(
    graphs: list[nx.DiGraph],
    labels: list[str],
    output_file: str = "comparison.html",
) -> list[str]:
    """One plot per graph, file names prefixed by the (sanitised) label."""
    if len(graphs) != len(labels):
        raise ValueError("graphs and labels must have the same length")
    out = []
    for graph, label in zip(graphs, labels):
        safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        out.append(plot_graph(graph, title=label, output_file=f"{safe}_{output_file}"))
    return out
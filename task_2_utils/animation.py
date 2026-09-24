"""Animated edit paths: the BIC-cherry network turning into the LRT.

    from task_2_utils.animation import trace_search, save_trace, animate

    result = compare_searches(N, T_star)["extended/prefer"]   # any SearchResult
    trace = trace_search(N, result, target=T_star)            # replay the path
    save_trace(trace, "results/trace.json")                   # store it
    animate(trace, "07_bic_to_lrt.html")                      # watch it

What is stored
--------------
A :class:`task_2_utils.search.SearchResult` already carries the move path. What it
does not carry is how to *replay* it, and that is not just "apply the moves":
the search merges twins once before it starts and again after every move
(``reduce_first`` / ``reduce_each``), and ``apply_move`` normalises (removes
dead vertices, suppresses single-child vertices). :func:`replay` repeats
exactly that, and :func:`trace_search` checks that the replay ends in
``result.network`` -- so the stored path provably reproduces the search.

A trace on disk (:func:`save_trace`) is the start network, the moves, the
target and the replay settings as JSON; the frames are recomputed on
:func:`load_trace`, and the stored end state is checked again.

Frames
------
One frame per state: the start network, the twin reduction (if it merged
anything), and for every move the network after the move -- followed by an
extra frame when the twin merging folded into that step merged something.
Every frame records whether it still explains the graph, its cost, whether it
is a tree and whether it *is* the target.

The HTML
--------
One self-contained file (no CDN, works offline), plain SVG + JavaScript:

* vertices keep their position between frames as long as they exist, and
  glide to the new one; appearing vertices/arcs fade in green, disappearing
  ones fade out red -- so every move is visible as what it is;
* leaves stay on the bottom row in a fixed order (the order of ``T*``), the
  root on top, inner vertices on the row of their depth;
* play / pause / step / scrub, speed, a clickable list of all steps with
  ✓/✗ for "still explains G", and ``T*`` as a thumbnail to compare against;
* keyboard: space = play/pause, ←/→ = step, Home/End.
"""

from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Iterable

import networkx as nx

from task_2_utils.graph_editing import Move, apply_move, reduce_twins, target_distance, tree_likeness
from utils.graph_utils import is_phylogenetic_tree, same_phylogeny
from utils.labels import class_names, leaf_labels
from utils.plotting import (
    INNER_GAP,
    LEAF_GAP,
    LEVEL_GAP,
    _clusters,
    _inner_x,
    _natural,
    _spread,
    layout as _plot_layout,
    node_color,
)

__all__ = [
    "Frame",
    "EditTrace",
    "replay",
    "trace_search",
    "trace_path",
    "save_trace",
    "load_trace",
    "animate",
    "animate_search",
]


# ---------------------------------------------------------------------------
# best match graphs (fast path when available)
# ---------------------------------------------------------------------------

def _bmg_arcs(N: nx.DiGraph, weak: bool) -> frozenset:
    try:
        from task_2_utils.fast_bmg import fast_bmg
        return frozenset(fast_bmg(N)[1 if weak else 0].edges())
    except ImportError:  # pragma: no cover
        from utils.graph_utils import bmg_from_network
        return frozenset(bmg_from_network(N, weak=weak).edges())


def _reticulations(N: nx.DiGraph) -> int:
    return sum(max(0, N.in_degree(v) - 1) for v in N)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    network: nx.DiGraph
    kind: str                 # "start" | "twins" | "move" | "merge"
    move: Move | None = None
    explains: bool = True
    cost: tuple = ()
    is_tree: bool = False
    at_target: bool = False


@dataclass
class EditTrace:
    start: nx.DiGraph
    moves: list
    target: nx.DiGraph | None = None
    weak: bool = False
    reduce_first: bool = True
    reduce_each: bool = True
    reference: frozenset = frozenset()
    frames: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final(self) -> nx.DiGraph:
        return self.frames[-1].network

    @property
    def reached_target(self) -> bool:
        return bool(self.frames) and self.frames[-1].at_target

    def summary(self) -> dict:
        return {
            "moves": len(self.moves),
            "frames": len(self.frames),
            "reached_target": self.reached_target,
            "always_explains": all(f.explains for f in self.frames),
            "non_explaining_frames": sum(not f.explains for f in self.frames),
            "start": [self.start.number_of_nodes(), self.start.number_of_edges()],
            "end": [self.final.number_of_nodes(), self.final.number_of_edges()],
        }


def _same_graph(a: nx.DiGraph, b: nx.DiGraph) -> bool:
    return set(a.nodes) == set(b.nodes) and set(a.edges) == set(b.edges)


def replay(
    N: nx.DiGraph,
    moves: Iterable[Move],
    reduce_first: bool = True,
    reduce_each: bool = True,
) -> list[tuple[str, nx.DiGraph, Move | None]]:
    """Re-run a move path exactly as :mod:`task_2_utils.search` ran it.

    Returns ``(kind, network, move)`` per frame. Raises ``ValueError`` if a
    move no longer applies (i.e. the path does not belong to ``N``).
    """
    leaves = {v for v in N if N.out_degree(v) == 0}
    frames = [("start", N.copy(), None)]
    state = N
    if reduce_first:
        reduced = reduce_twins(state)
        if not _same_graph(reduced, state):
            frames.append(("twins", reduced, None))
        state = reduced
    for i, move in enumerate(moves):
        move = Move(*move) if not isinstance(move, Move) else move
        try:
            M = apply_move(state, move, leaves)
        except (ValueError, nx.NetworkXError) as exc:
            raise ValueError(f"move {i} ({move}) does not apply: {exc}") from exc
        frames.append(("move", M, move))
        if reduce_each:
            R = reduce_twins(M)
            if not _same_graph(R, M):
                frames.append(("merge", R, move))
            M = R
        state = M
    return frames


def trace_path(
    N: nx.DiGraph,
    moves: Iterable[Move],
    target: nx.DiGraph | None = None,
    weak: bool = False,
    reduce_first: bool = True,
    reduce_each: bool = True,
    meta: dict | None = None,
) -> EditTrace:
    """Replay ``moves`` on ``N`` and annotate every frame."""
    moves = [Move(*m) if not isinstance(m, Move) else m for m in moves]
    reference = _bmg_arcs(N, weak)
    cost_of = (lambda M: target_distance(M, target)) if target is not None else tree_likeness
    trace = EditTrace(
        start=N.copy(), moves=moves, target=target, weak=weak,
        reduce_first=reduce_first, reduce_each=reduce_each,
        reference=reference, meta=dict(meta or {}),
    )
    for kind, M, move in replay(N, moves, reduce_first, reduce_each):
        tree = is_phylogenetic_tree(M)
        trace.frames.append(Frame(
            network=M,
            kind=kind,
            move=move,
            explains=_bmg_arcs(M, weak) == reference,
            cost=tuple(cost_of(M)),
            is_tree=tree,
            at_target=bool(target is not None and tree and same_phylogeny(M, target)),
        ))
    return trace


def trace_search(
    N: nx.DiGraph,
    result,
    target: nx.DiGraph | None = None,
    weak: bool = False,
    reduce_first: bool = True,
    reduce_each: bool = True,
) -> EditTrace:
    """:class:`EditTrace` of a :class:`task_2_utils.search.SearchResult` found on ``N``.

    ``weak`` / ``reduce_first`` / ``reduce_each`` must be what the search
    ran with (the defaults are the search defaults). The replay is checked
    against ``result.network``; a mismatch raises ``ValueError`` instead of
    animating something the search never did.
    """
    trace = trace_path(
        N, result.path, target=target, weak=weak,
        reduce_first=reduce_first, reduce_each=reduce_each,
        meta={"strategy": getattr(result, "strategy", "?"),
              "reached_target": bool(getattr(result, "reached_target", False))},
    )
    if not _same_graph(trace.final, result.network):
        raise ValueError(
            "replaying result.path does not reproduce result.network -- "
            "were weak/reduce_first/reduce_each the same as in the search?"
        )
    return trace


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def _enc(v) -> list:
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise TypeError(f"vertex id {v!r}: only int and str ids can be stored")
    return ["i", v] if isinstance(v, int) else ["s", v]


def _dec(item):
    tag, v = item
    return int(v) if tag == "i" else str(v)


def _graph_to_json(G: nx.DiGraph | None):
    if G is None:
        return None
    return {
        "nodes": [[_enc(v), G.nodes[v].get("color"), G.nodes[v].get("label")] for v in G],
        "edges": [[_enc(u), _enc(v)] for u, v in G.edges()],
    }


def _graph_from_json(data) -> nx.DiGraph | None:
    if data is None:
        return None
    G = nx.DiGraph()
    for v, color, label in data["nodes"]:
        G.add_node(_dec(v), color=color)
        if label is not None:
            G.nodes[_dec(v)]["label"] = label
    G.add_edges_from((_dec(u), _dec(v)) for u, v in data["edges"])
    return G


def save_trace(trace: EditTrace, path: str | Path) -> Path:
    """Write ``trace`` as JSON (start, moves, target, settings, end state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "bmg-edit-trace/1",
        "weak": trace.weak,
        "reduce_first": trace.reduce_first,
        "reduce_each": trace.reduce_each,
        "meta": trace.meta,
        "start": _graph_to_json(trace.start),
        "target": _graph_to_json(trace.target),
        "moves": [[m.kind, [_enc(a) for a in m.args]] for m in trace.moves],
        "final_edges": sorted([[_enc(u), _enc(v)] for u, v in trace.final.edges()]),
        "summary": trace.summary(),
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_trace(path: str | Path) -> EditTrace:
    """Read a trace written by :func:`save_trace` and replay it (checked)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    moves = [Move(kind, tuple(_dec(a) for a in args)) for kind, args in data["moves"]]
    trace = trace_path(
        _graph_from_json(data["start"]), moves,
        target=_graph_from_json(data["target"]), weak=data["weak"],
        reduce_first=data["reduce_first"], reduce_each=data["reduce_each"],
        meta=data.get("meta", {}),
    )
    stored = {(_dec(u), _dec(v)) for u, v in data["final_edges"]}
    if set(trace.final.edges()) != stored:
        raise ValueError("the stored path no longer reproduces the stored end state")
    return trace


# ---------------------------------------------------------------------------
# layout: stable positions across frames
# ---------------------------------------------------------------------------

def _levels(G: nx.DiGraph) -> dict:
    level = {v: 0 for v in G}
    for v in nx.topological_sort(G):
        for c in G.successors(v):
            level[c] = max(level[c], level[v] + 1)
    return level


def _leaf_order(trace: EditTrace, names: dict, classes: dict) -> list:
    """Leaves in the left-to-right order of ``T*`` (or of the final frame)."""
    ref = trace.target if trace.target is not None else trace.final
    pos = _plot_layout(ref, "tree" if is_phylogenetic_tree(ref) else "network")
    leaves = [v for v in trace.start if trace.start.out_degree(v) == 0]
    return sorted(leaves, key=lambda v: (pos[v][0] if v in pos else 0, _natural(names.get(v, v))))


def _frame_positions(G, leaf_x, height) -> dict:
    """Leaves at fixed x on the bottom row, inner vertices by depth/barycentre."""
    level = _levels(G)
    depth = max((level[v] for v in G if G.out_degree(v) == 0), default=1) or 1
    order = list(nx.topological_sort(G))
    x = _inner_x(G, order, level, leaf_x, tree_like=False, gap=INNER_GAP)
    pos = {}
    for v in G:
        if G.out_degree(v) == 0:
            pos[v] = (leaf_x[v], height)
        else:
            pos[v] = (x[v], height * min(level[v], depth - 1) / depth)
    # one more pass: rows can collide after the depth rescaling
    rows: dict = {}
    for v, (_, y) in pos.items():
        if G.out_degree(v) > 0:
            rows.setdefault(round(y, 3), []).append(v)
    xs = {v: p[0] for v, p in pos.items()}
    for row in rows.values():
        _spread(row, xs, INNER_GAP)
    return {v: (xs[v], pos[v][1]) for v in pos}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _display(v, names: dict) -> str:
    """``p:12|4`` -> ``p:b1|a1`` (vertex ids inside inner names -> leaf names)."""
    if v in names:
        return str(names[v])
    s = str(v)
    m = re.match(r"^([pq]):(.+)\|(.+)$", s)
    if m:
        def nm(tok):
            for leaf, name in names.items():
                if str(leaf) == tok:
                    return str(name)
            return tok
        return f"{m.group(1)}:{nm(m.group(2))}|{nm(m.group(3))}"
    return s


def _describe(frame: Frame, names: dict) -> str:
    d = lambda v: _display(v, names)  # noqa: E731
    if frame.kind == "start":
        return "BIC-cherry network (start)"
    if frame.kind == "twins":
        return "merge twin vertices (p:x|y = p:y|x) — preserves the BMG"
    if frame.kind == "merge":
        return "…and merge the twins this created"
    m = frame.move
    a = m.args
    if m.kind == "pull_up":
        return f"pull up  {d(a[0])}→{d(a[1])}  to  {d(a[2])}→{d(a[1])}"
    if m.kind == "pull_down":
        return f"pull down  {d(a[0])}→{d(a[1])}  to  {d(a[2])}→{d(a[1])}"
    if m.kind == "merge_twins":
        return f"merge twins  {d(a[1])} into {d(a[0])}"
    if m.kind == "remove_arc":
        return f"remove arc  {d(a[0])}→{d(a[1])}"
    return f"{m.kind} {tuple(d(x) for x in a)}"


def _node_payload(G, v, xy, names, classes) -> dict:
    leaf = G.out_degree(v) == 0
    role = ("leaf" if leaf else "root" if G.in_degree(v) == 0 else
            "cherry" if str(v).startswith("p:") else
            "extension" if str(v).startswith("q:") else "inner")
    return {
        "id": repr(v),
        "label": _display(v, names),
        "sub": classes.get(v, "") if leaf else "",
        "x": round(xy[0], 2),
        "y": round(xy[1], 2),
        "color": node_color(G, v),
        "leaf": leaf,
        "role": role,
    }


def _payload(trace: EditTrace, title: str) -> dict:
    names = leaf_labels(trace.start)
    classes = class_names(trace.start, names)
    leaves = _leaf_order(trace, names, classes)

    widest = 1
    for f in trace.frames:
        rows: dict = {}
        for v, lvl in _levels(f.network).items():
            if f.network.out_degree(v) > 0:
                rows[lvl] = rows.get(lvl, 0) + 1
        widest = max([widest, *rows.values()])
    width = max(len(leaves) * LEAF_GAP, widest * INNER_GAP)
    gap = width / max(1, len(leaves))
    leaf_x = {v: (i - (len(leaves) - 1) / 2) * gap for i, v in enumerate(leaves)}
    max_depth = max(max(_levels(f.network).values(), default=1) for f in trace.frames)
    height = max(LEVEL_GAP * 3, min(LEVEL_GAP * max_depth, 0.55 * width))

    frames = []
    step = 0
    for f in trace.frames:
        if f.kind == "move":
            step += 1
        G = f.network
        pos = _frame_positions(G, leaf_x, height)
        frames.append({
            "step": step,
            "kind": f.kind,
            "caption": _describe(f, names),
            "explains": f.explains,
            "is_tree": f.is_tree,
            "at_target": f.at_target,
            "cost": list(f.cost),
            "V": G.number_of_nodes(),
            "E": G.number_of_edges(),
            "retic": _reticulations(G),
            "nodes": [_node_payload(G, v, pos[v], names, classes)
                      for v in sorted(G, key=lambda v: _natural(repr(v)))],
            "edges": sorted([repr(u), repr(v)] for u, v in G.edges()),
        })

    target = None
    if trace.target is not None:
        T = trace.target
        tpos = _frame_positions(T, leaf_x, height)
        target = {
            "nodes": [_node_payload(T, v, tpos[v], names, classes) for v in T],
            "edges": [[repr(u), repr(v)] for u, v in T.edges()],
        }

    legend = {}
    for v, c in classes.items():
        legend.setdefault(c, []).append(str(names[v]))
    return {
        "title": title,
        "graph": "WBMG" if trace.weak else "BMG",
        "frames": frames,
        "target": target,
        "classes": [[c, sorted(m, key=_natural)] for c, m in sorted(legend.items(), key=lambda kv: _natural(kv[0]))],
        "summary": trace.summary(),
    }


def animate(
    trace: EditTrace,
    output_file: str | Path = "edit_path.html",
    title: str | None = None,
    seconds_per_step: float = 1.2,
) -> str:
    """Write the animation of ``trace`` as one self-contained HTML file."""
    s = trace.summary()
    title = title or (
        f"BIC-cherry network → T*  ·  {s['moves']} moves  ·  "
        + ("reached T*" if trace.reached_target else "did not reach T*")
    )
    data = _payload(trace, title)
    page = (
        _TEMPLATE
        .replace("__TITLE__", _html.escape(title))
        .replace("__STEP_MS__", str(int(seconds_per_step * 1000)))
        .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(page, encoding="utf-8")
    print(f"✓ animation ({len(trace.frames)} frames) saved to {output_file}")
    return str(output_file)


def animate_search(N, result, target=None, output_file="edit_path.html", **kwargs) -> EditTrace:
    """:func:`trace_search` + :func:`animate` in one call; returns the trace."""
    trace = trace_search(N, result, target=target)
    animate(trace, output_file, **kwargs)
    return trace


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #fafaf8; --panel: #ffffff; --ink: #1f2328; --muted: #6a7078; --line: #e3e3de;
  --edge: #8a8f96; --add: #2e9e5b; --del: #d0453a; --ok: #2e9e5b; --bad: #d0453a; --accent: #3a6ea5;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #16181b; --panel: #1e2125; --ink: #e8e9ea; --muted: #9aa0a6; --line: #30343a;
          --edge: #8d939b; --add: #4cc47f; --del: #ef6b5f; --ok: #4cc47f; --bad: #ef6b5f; --accent: #7aa7da; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 14px 20px 6px; }
h1 { font-size: 17px; margin: 0 0 2px; font-weight: 600; }
.sub { color: var(--muted); font-size: 13px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; padding: 8px 20px 20px; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
#stage { position: relative; overflow: hidden; }
#svg { width: 100%; height: 62vh; min-height: 360px; display: block; }
#caption { padding: 10px 14px; border-top: 1px solid var(--line); display: flex; gap: 12px;
           align-items: center; flex-wrap: wrap; min-height: 44px; }
#caption .text { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.pill.ok { color: var(--ok); border-color: var(--ok); }
.pill.bad { color: var(--bad); border-color: var(--bad); }
.pill.goal { color: #fff; background: var(--ok); border-color: var(--ok); }
#controls { display: flex; gap: 6px; align-items: center; padding: 10px 14px; border-top: 1px solid var(--line); flex-wrap: wrap; }
button, select { font: inherit; background: var(--panel); color: var(--ink); border: 1px solid var(--line);
                 border-radius: 7px; padding: 5px 10px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); min-width: 70px; }
#scrub { flex: 1; min-width: 120px; accent-color: var(--accent); }
#pos { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 72px; text-align: right; }
label.tog { color: var(--muted); font-size: 13px; display: flex; gap: 4px; align-items: center; }
aside { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
aside h2 { font-size: 13px; margin: 0; padding: 10px 12px 6px; color: var(--muted); font-weight: 600;
           text-transform: uppercase; letter-spacing: .04em; }
#stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--line);
         border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; overflow: hidden; }
#stats div { background: var(--panel); padding: 8px 10px; }
#stats b { display: block; font-size: 17px; font-variant-numeric: tabular-nums; }
#stats span { color: var(--muted); font-size: 12px; }
#thumb { width: 100%; height: 170px; display: block; border-top: 1px solid var(--line); }
#steps { list-style: none; margin: 0; padding: 0 0 6px; max-height: 38vh; overflow: auto; border-top: 1px solid var(--line); }
#steps li { display: grid; grid-template-columns: 34px 18px 1fr; gap: 4px; padding: 5px 10px; cursor: pointer;
            font-size: 12.5px; border-left: 3px solid transparent; }
#steps li:hover { background: color-mix(in srgb, var(--accent) 8%, transparent); }
#steps li.cur { border-left-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
#steps .n { color: var(--muted); font-variant-numeric: tabular-nums; }
#steps .m { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
.okc { color: var(--ok); } .badc { color: var(--bad); }
#legend { padding: 0 12px 10px; font-size: 12.5px; color: var(--muted); }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
.nlab { font-size: 13px; text-anchor: middle; }
.ilab { font-size: 9.5px; fill: var(--muted); }
.slab { font-size: 11px; text-anchor: middle; fill: var(--muted); }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="subtitle"></div>
</header>
<main>
  <section class="card" id="stage">
    <svg id="svg" role="img" aria-label="edit path animation"></svg>
    <div id="caption"><span class="text" id="cap"></span><span id="badges"></span></div>
    <div id="controls">
      <button id="first" title="first (Home)">⏮</button>
      <button id="prev" title="previous (←)">◀</button>
      <button id="play" class="primary" title="play / pause (space)">▶ play</button>
      <button id="next" title="next (→)">▶</button>
      <button id="last" title="last (End)">⏭</button>
      <input type="range" id="scrub" min="0" value="0" aria-label="frame">
      <span id="pos"></span>
      <select id="speed" title="speed">
        <option value="0.5">0.5×</option><option value="1" selected>1×</option>
        <option value="2">2×</option><option value="4">4×</option>
      </select>
      <label class="tog"><input type="checkbox" id="inner"> inner labels</label>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>Current state</h2>
      <div id="stats"></div>
    </div>
    <div class="card" id="targetcard">
      <h2>Target T*</h2>
      <svg id="thumb" role="img" aria-label="target least resolved tree"></svg>
      <div id="legend"></div>
    </div>
    <div class="card">
      <h2>Steps</h2>
      <ol id="steps"></ol>
    </div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const STEP_MS = __STEP_MS__;
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const F = DATA.frames;

$("title").textContent = DATA.title;
$("subtitle").textContent =
  `${F.length} frames · every frame is checked against the ${DATA.graph} it has to explain` +
  (DATA.summary.non_explaining_frames ? ` · ${DATA.summary.non_explaining_frames} frames do NOT explain it (✗)` : " · all frames explain it");

/* ---------- geometry ---------- */
let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
for (const f of F) for (const n of f.nodes) {
  minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
}
const pad = 60;
const svg = $("svg");
const vbW = maxX - minX + 2 * pad;
// wide networks (hundreds of px per row) would shrink nodes and text to
// nothing -- scale them with the drawing instead
const S = Math.max(1, vbW / Math.max(1, svg.clientWidth || 900));
svg.setAttribute("viewBox", `${minX - pad} ${minY - pad * S} ${vbW} ${maxY - minY + pad + 70 * S}`);
const defs = document.createElementNS(NS, "defs");
for (const [key, colVar] of [["e", "--edge"], ["add", "--add"], ["del", "--del"]]) {
  const m = document.createElementNS(NS, "marker");
  m.setAttribute("id", "arr-" + key); m.setAttribute("viewBox", "0 0 10 10");
  m.setAttribute("refX", "9"); m.setAttribute("refY", "5");
  m.setAttribute("markerWidth", "7"); m.setAttribute("markerHeight", "7"); m.setAttribute("orient", "auto-start-reverse");
  const p = document.createElementNS(NS, "path"); p.setAttribute("d", "M0,0 L10,5 L0,10 z");
  p.setAttribute("fill", css(colVar)); m.appendChild(p); defs.appendChild(m);
}
svg.appendChild(defs);
const gEdges = document.createElementNS(NS, "g"), gNodes = document.createElementNS(NS, "g");
svg.appendChild(gEdges); svg.appendChild(gNodes);

const radius = n => (n.leaf ? 12 : n.role === "root" ? 8 : 6) * S;

/* ---------- persistent elements ---------- */
const nodeEls = new Map(), edgeEls = new Map();
function nodeEl(n) {
  let el = nodeEls.get(n.id);
  if (el) return el;
  const g = document.createElementNS(NS, "g");
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("r", radius(n)); c.setAttribute("fill", n.color);
  c.setAttribute("stroke", "rgba(0,0,0,.35)"); c.setAttribute("stroke-width", S);
  const halo = document.createElementNS(NS, "circle");
  halo.setAttribute("r", radius(n) + 6 * S); halo.setAttribute("fill", "none"); halo.setAttribute("stroke-width", 3 * S);
  halo.setAttribute("opacity", "0");
  const t = document.createElementNS(NS, "text");
  const title = document.createElementNS(NS, "title");
  title.textContent = `${n.label}  (${n.role}, id ${n.id})`;
  if (n.leaf) {
    t.setAttribute("class", "nlab"); t.setAttribute("y", 30 * S); t.style.fontSize = 13 * S + "px"; t.textContent = n.label;
    g.appendChild(halo); g.appendChild(c); g.appendChild(t);
    if (n.sub) { const s = document.createElementNS(NS, "text"); s.setAttribute("class", "slab"); s.setAttribute("y", 45 * S);
      s.style.fontSize = 11 * S + "px"; s.textContent = n.sub; g.appendChild(s); }
  } else {
    t.setAttribute("class", "ilab inner"); t.setAttribute("x", 9 * S); t.setAttribute("y", -7 * S);
    t.style.fontSize = 9.5 * S + "px"; t.textContent = n.label;
    g.appendChild(halo); g.appendChild(c); g.appendChild(t);
  }
  g.appendChild(title);
  el = { g, halo, x: n.x, y: n.y, r: radius(n), op: 0 };
  gNodes.appendChild(g); nodeEls.set(n.id, el);
  return el;
}
function edgeEl(key) {
  let el = edgeEls.get(key);
  if (el) return el;
  const l = document.createElementNS(NS, "line");
  l.setAttribute("stroke-width", 1.4 * S);
  gEdges.appendChild(l);
  el = { l, op: 0 };
  edgeEls.set(key, el); return el;
}
function place(el) { el.g.setAttribute("transform", `translate(${el.x},${el.y})`); el.g.setAttribute("opacity", el.op); }
function drawEdge(el, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
  el.l.setAttribute("x1", a.x + dx / d * (a.r + 1)); el.l.setAttribute("y1", a.y + dy / d * (a.r + 1));
  el.l.setAttribute("x2", b.x - dx / d * (b.r + 3 * S)); el.l.setAttribute("y2", b.y - dy / d * (b.r + 3 * S));
  el.l.setAttribute("opacity", el.op);
}

/* ---------- thumbnail of T* ---------- */
if (DATA.target) {
  const th = $("thumb"); const T = DATA.target;
  const xs = T.nodes.map(n => n.x), ys = T.nodes.map(n => n.y);
  const tw = Math.max(...xs) - Math.min(...xs) + 60;
  const k = Math.max(1, tw / Math.max(1, th.clientWidth || 300));
  const x0 = Math.min(...xs) - 30 * k, y0 = Math.min(...ys) - 14 * k;
  th.setAttribute("viewBox", `${x0} ${y0} ${Math.max(...xs) - x0 + 30 * k} ${Math.max(...ys) - y0 + 30 * k}`);
  const at = Object.fromEntries(T.nodes.map(n => [n.id, n]));
  for (const [u, v] of T.edges) {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", at[u].x); l.setAttribute("y1", at[u].y); l.setAttribute("x2", at[v].x); l.setAttribute("y2", at[v].y);
    l.setAttribute("stroke", css("--edge")); l.setAttribute("stroke-width", 1.5 * k); th.appendChild(l);
  }
  for (const n of T.nodes) {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", n.x); c.setAttribute("cy", n.y); c.setAttribute("r", (n.leaf ? 7 : 4) * k); c.setAttribute("fill", n.color);
    th.appendChild(c);
    if (n.leaf) { const t = document.createElementNS(NS, "text"); t.setAttribute("x", n.x); t.setAttribute("y", n.y + 22 * k);
      t.setAttribute("class", "nlab"); t.style.fontSize = 12 * k + "px"; t.textContent = n.label; th.appendChild(t); }
  }
} else { $("targetcard").style.display = "none"; }
if (DATA.classes.length)
  $("legend").textContent = "thinness classes: " + DATA.classes.map(([c, m]) => `${c} = {${m.join(", ")}}`).join(" · ");

/* ---------- step list ---------- */
const list = $("steps");
F.forEach((f, i) => {
  const li = document.createElement("li");
  li.innerHTML = `<span class="n">${f.kind === "move" ? f.step : ""}</span>` +
    `<span class="${f.explains ? "okc" : "badc"}">${f.explains ? "✓" : "✗"}</span><span class="m"></span>`;
  li.querySelector(".m").textContent = f.caption + (f.at_target ? "  = T*" : "");
  li.onclick = () => { stop(); go(i, true); };
  list.appendChild(li);
});

/* ---------- state + transitions ---------- */
let cur = 0, playing = false, timer = null, anim = null;
const speed = () => parseFloat($("speed").value);

function badges(f) {
  const b = [];
  b.push(f.explains ? `<span class="pill ok">explains ${DATA.graph} ✓</span>` : `<span class="pill bad">does not explain ${DATA.graph} ✗</span>`);
  if (f.at_target) b.push(`<span class="pill goal">= T*</span>`);
  else if (f.is_tree) b.push(`<span class="pill">tree</span>`);
  return b.join(" ");
}
function info(i) {
  const f = F[i];
  $("cap").textContent = (f.kind === "move" ? `step ${f.step}: ` : "") + f.caption;
  $("badges").innerHTML = badges(f);
  $("pos").textContent = `${i + 1} / ${F.length}`;
  $("scrub").value = i;
  const moves = F.filter(x => x.kind === "move").length;
  $("stats").innerHTML =
    `<div><b>${f.step}</b><span>of ${moves} moves</span></div>` +
    `<div><b>${f.V}</b><span>vertices</span></div>` +
    `<div><b>${f.E}</b><span>arcs</span></div>` +
    `<div><b>${f.retic}</b><span>reticulations</span></div>` +
    `<div><b>${f.cost.length ? f.cost[0] : "–"}</b><span>cost to T*</span></div>` +
    `<div><b class="${f.explains ? "okc" : "badc"}">${f.explains ? "✓" : "✗"}</b><span>explains ${DATA.graph}</span></div>`;
  [...list.children].forEach((li, k) => li.classList.toggle("cur", k === i));
  const li = list.children[i];
  if (li) li.scrollIntoView({ block: "nearest" });
}

const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
function go(i, instant = false) {
  i = Math.max(0, Math.min(F.length - 1, i));
  const to = F[i];
  const nodesTo = new Map(to.nodes.map(n => [n.id, n]));
  const edgesTo = new Set(to.edges.map(([u, v]) => u + "→" + v));
  const add = css("--add"), del = css("--del"), grey = css("--edge");
  const start = new Map();
  for (const n of to.nodes) { const el = nodeEl(n); start.set(n.id, { x: el.x, y: el.y, op: el.op }); }
  for (const [id, el] of nodeEls) if (!nodesTo.has(id)) start.set(id, { x: el.x, y: el.y, op: el.op });
  for (const [u, v] of to.edges) edgeEl(u + "→" + v);
  const eStart = new Map([...edgeEls].map(([k, el]) => [k, el.op]));

  // colour the changes
  for (const [k, el] of edgeEls) {
    const born = edgesTo.has(k) && el.op < 0.99, dying = !edgesTo.has(k) && el.op > 0.01;
    const c = born ? add : dying ? del : grey;
    el.l.setAttribute("stroke", c);
    el.l.setAttribute("stroke-dasharray", dying ? "5 4" : "");
    el.l.setAttribute("marker-end", `url(#arr-${born ? "add" : dying ? "del" : "e"})`);
  }
  for (const [id, el] of nodeEls) {
    const born = nodesTo.has(id) && el.op < 0.99, dying = !nodesTo.has(id) && el.op > 0.01;
    el.halo.setAttribute("stroke", born ? add : del);
    el.halo.setAttribute("opacity", born || dying ? 0.9 : 0);
    if (born) { const n = nodesTo.get(id); el.x = n.x; el.y = n.y; start.set(id, { x: n.x, y: n.y, op: 0 }); }
  }

  const dur = instant ? 0 : STEP_MS * 0.75 / speed();
  const t0 = performance.now();
  if (anim) cancelAnimationFrame(anim);
  const frame = now => {
    const t = dur ? Math.min(1, (now - t0) / dur) : 1, e = ease(t);
    for (const [id, el] of nodeEls) {
      const s = start.get(id) || { x: el.x, y: el.y, op: el.op };
      const n = nodesTo.get(id);
      const tx = n ? n.x : s.x, ty = n ? n.y : s.y, to_op = n ? 1 : 0;
      el.x = s.x + (tx - s.x) * e; el.y = s.y + (ty - s.y) * e; el.op = s.op + (to_op - s.op) * e;
      place(el);
    }
    for (const [k, el] of edgeEls) {
      const [u, v] = k.split("→");
      const s = eStart.get(k) ?? 0, target = edgesTo.has(k) ? 1 : 0;
      el.op = s + (target - s) * e;
      const a = nodeEls.get(u), b = nodeEls.get(v);
      if (a && b) drawEdge(el, a, b);
    }
    if (t < 1) anim = requestAnimationFrame(frame);
    else {
      anim = null;
      for (const [k, el] of edgeEls) {           // settle: new arcs turn grey
        if (edgesTo.has(k)) { el.l.setAttribute("stroke", grey); el.l.setAttribute("marker-end", "url(#arr-e)"); }
      }
      for (const [, el] of nodeEls) el.halo.setAttribute("opacity", 0);
    }
  };
  anim = requestAnimationFrame(frame);
  cur = i; info(i);
}

function stop() { playing = false; clearTimeout(timer); $("play").textContent = "▶ play"; }
function tick() {
  if (!playing) return;
  if (cur >= F.length - 1) { stop(); return; }
  go(cur + 1);
  timer = setTimeout(tick, STEP_MS / speed());
}
function play() {
  if (playing) { stop(); return; }
  if (cur >= F.length - 1) go(0, true);
  playing = true; $("play").textContent = "⏸ pause";
  timer = setTimeout(tick, 250);
}
$("play").onclick = play;
$("next").onclick = () => { stop(); go(cur + 1); };
$("prev").onclick = () => { stop(); go(cur - 1); };
$("first").onclick = () => { stop(); go(0); };
$("last").onclick = () => { stop(); go(F.length - 1); };
$("scrub").max = F.length - 1;
$("scrub").oninput = e => { stop(); go(+e.target.value, true); };
const innerBox = $("inner");
const innerInner = F[0].nodes.filter(n => !n.leaf).length;
innerBox.checked = innerInner <= 24;
const applyInner = () => document.querySelectorAll(".inner").forEach(t => t.style.display = innerBox.checked ? "" : "none");
innerBox.onchange = applyInner;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "ArrowRight") { stop(); go(cur + 1); }
  else if (e.key === "ArrowLeft") { stop(); go(cur - 1); }
  else if (e.key === "Home") { stop(); go(0); }
  else if (e.key === "End") { stop(); go(F.length - 1); }
});
go(0, true);
// labels of vertices created later also follow the toggle
new MutationObserver(applyInner).observe(gNodes, { childList: true });
applyInner();
</script>
</body>
</html>
"""

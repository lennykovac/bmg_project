# BMG Project für Fortgeschrittene Methoden

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In a leaf-colored phylogenetic network `(N, σ)` the colors are species and the
leaves are genes. A gene `y` is a **best match** of `x` if `lca(x, y)` is as low
as an lca of `x` with a gene of `y`'s species ever gets; it is a **weak best
match** if `lca(x, y)` meets one of those lowest common ancestors, even when
another element of `lca(x, y)` sits higher up. On trees both definitions agree
— they only come apart on genuine networks, where an lca can be a set.

This project compares the two definitions and studies the networks that explain
such graphs. Instances are simulated: [AsymmeTree](https://github.com/david-schaller/AsymmeTree)
produces a dated gene tree, and `transform` inserts random hybridization
vertices into it to obtain a network.

## 🚀 Objectives

1. **Compare best matches and weak best matches.**
   1. Check whether every best match in `(N, σ)` is also a weak best match.
      Test this computationally, but also try to give a formal argument.
   2. Check if a modified version of the BIC-cherry + expansion procedure
      yields networks that explain weak best match graphs.
2. **Find a way to edit the explaining networks** for (weak) best matches to
   become more tree like.

Objective 1 is what `main.py` runs: `1a` for the first item, and `1b` and `1d`
for two different constructions answering the second one. Objective 2 lives in
`task_2_utils/` and is driven from its own modules and tests, not from the
entry point.

## Getting started

Everything runs through [uv](https://docs.astral.sh/uv/guides/scripts/)

```sh
uv run main.py        # every experiment of objective 1
uv run main.py 1a     # only 1a: best matches vs weak best matches
uv run main.py 1b     # only 1b: cherries for reciprocal edges + witness chains
uv run main.py 1d     # only 1d: cherries + extensions found by backtracking
uv run main.py 1b 1d  # the two constructions on the same instances
```

The run is seeded (`SEED` in [`main.py`](main.py)), so the numbers are
reproducible. The exit status is non-zero if an experiment contradicts the
theory, which makes the script usable as a check.

### What the experiments do

**1a — best matches vs weak best matches.** For every simulated network the BMG
and the WBMG are computed and compared as edge sets, in both directions. The
formal argument predicts the inclusion `BMG ⊆ WBMG` but not the converse, so
every network is a test of that claim; a violation is reported with the weak
best matches it is missing.

```
objective 1a: best matches vs weak best matches (200 networks, ~12 leaves, 3 species, ~4 hybrids)
    best match => weak best match       200  100.0%
    weak best match => best match       153   76.5%
    BMG = WBMG                          153   76.5%

    no counterexample: every best match was also a weak best match
```

**1b — a BIC-cherry construction for weak best matches.** Plain BIC-cherry +
expansion does not work for weak best matches, because every cherry vertex is a
lowest common ancestor for both directions of its pair. `wbmg_cherry_extension`
gives a cherry only to reciprocal edges and explains a one-directional edge by
extending along a witness chain. Each instance is a round trip
`WBMG → network → WBMG'`, which is correct exactly if `WBMG'` equals the graph
the network was built from.

```
objective 1b: BIC-cherry + extension for weak best matches (100 networks, ~10 leaves, 2 species, ~3 hybrids)
    WBMG of the construction = WBMG     100  100.0%
    different WBMG                        0    0.0%
    graph refused as no WBMG              0    0.0%

    the construction explained every weak best match graph it accepted
```

**1d — the same question, answered by search.** [`bic_cherry_wmb.py`](bic_cherry_wmb.py)
takes the other route. It keeps the plain cherry network and suppresses a false
edge `x → y` by hanging an extension vertex `q:x|z` below the cherry, which
makes `x` prefer `z`. The partner `z` is picked by a depth first search that
tries the most promising candidates first and fixes the false edges the new
vertex creates below it; on a dead end it removes what it added and tries the
next partner. A false edge that no partner removes is only warned about, not
raised, so those dead ends get a row of their own. The parameters are the ones
of 1b, so both constructions see the same instances and the tables can be
compared directly.

```
objective 1d: BIC-cherry + expansion with backtracking (100 networks, ~10 leaves, 2 species, ~3 hybrids)
    WBMG of the construction = WBMG     100  100.0%
    different WBMG                        0    0.0%
    search ran into a dead end            0    0.0%

    the backtracking search explained every weak best match graph
```

### Larger runs

The sample sizes are the keyword defaults of the three functions, so a bigger
run is one import away:

```sh
uv run python -c "from main import objective_1a; objective_1a(runs=1000, leaves=20, species=4, hybrids=6)"
```

They take a `seed` as well and seed the simulation themselves, so a single call
reproduces on its own and `uv run main.py 1b` prints the same numbers whether it
runs alone or as part of the whole script.

## Tests

```sh
uv run pytest                 # the normal suite
uv run pytest --exhaustive    # plus the compute-intensive tests
```

## Repository layout

| Path | Contents |
| --- | --- |
| [`main.py`](main.py) | entry point for objective 1 |
| [`bic_cherry_wmb.py`](bic_cherry_wmb.py) | the backtracking construction for weak best match graphs (1d) |
| [`utils/`](utils/) | shared code: (weak) best match graphs, least resolved trees, BIC-cherry constructions, tree simulation, plotting |
| [`task_2_utils/`](task_2_utils/) | objective 2: editing explaining networks towards the least resolved tree |
| [`tasks/`](tasks/) | experiment scripts written while working on the objectives |
| [`tests/`](tests/) | pytest suite, including the round-trip tests of the constructions |
| [`docs/`](docs/) | the project description and the papers it refers to |

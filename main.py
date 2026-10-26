"""Entry point for objective 1: best matches vs weak best matches.

Both experiments run on the same kind of instance. AsymmeTree simulates a
dated gene tree, and ``utils.graph_utils.transform`` inserts random
hybridization vertices into it, which turns the tree into a phylogenetic
network (N, sigma).

objective 1a -- is every best match of (N, sigma) also a weak best match?
    For every network the BMG and the WBMG are computed and compared as edge
    sets, in both directions. The formal argument predicts the inclusion
    BMG <= WBMG but not the converse, so a network violating it would be a
    counterexample and is printed with its missing weak best matches.

objective 1b -- does a modified BIC-cherry + expansion explain weak best match
    graphs? Round trip per network: WBMG -> ``utils.bic_cherry``'s
    ``wbmg_cherry_extension`` -> N' -> WBMG'. The construction is right on
    that instance exactly if WBMG' equals the WBMG it was built from.

objective 1d -- the same question for the other construction,
    ``bic_cherry_wmb``'s ``bic_cherry_for_wbm``. It keeps the plain cherry
    network and suppresses every false edge by a depth first search with
    backtracking over the possible extension partners. A false edge that no
    partner removes is a dead end, which the construction warns about instead
    of raising, so those are counted as well.

invocation (see README.md):
    uv run main.py          run every objective
    uv run main.py 1a       run objective 1a only
    uv run main.py 1b 1d    compare the two constructions

The sample sizes are the keyword defaults of the functions below, so a bigger
run is one import away:

    from main import objective_1a
    objective_1a(runs=1000, leaves=20, species=4, hybrids=6)
"""

import random
import sys
import warnings

import networkx as nx
import numpy as np

from bic_cherry_wmb import bic_cherry_for_wbm
from utils.bic_cherry import wbmg_cherry_extension
from utils.graph_utils import bmg_from_network, transform, wbmg_from_network
from utils.tree_utils import create_gene_tree_n_leaves

SEED = 1


def seed_rngs(seed: int) -> None:
    """Seed both generators asymmetree draws from, so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def random_network(leaves: int, species: int, hybrids: int) -> nx.DiGraph:
    """
    A simulated gene tree with roughly ``leaves`` leaves, as a network.

    Parameters:
    leaves: number of leaves of the gene tree
    species: number of species in the species tree (= number of colors)
    hybrids: number of hybridization vertices to insert

    Returns:
    nx.DiGraph of a phylogenetic network with colored leaves.
    """
    tree = create_gene_tree_n_leaves(leaves, species).gene_tree
    return transform(tree, hybrids)


def result_line(label: str, count: int, total: int) -> str:
    """One result line of a summary: label, count and share of the runs."""
    share = f"{100 * count / total:.1f}%" if total else "-"
    return f"    {label:<33}{count:>6}{share:>8}"


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def objective_1a(runs: int = 200, leaves: int = 12, species: int = 3,
                 hybrids: int = 4, seed: int = SEED) -> bool:
    """
    Compare best matches and weak best matches on random networks.

    Parameters:
    runs: number of networks to test
    leaves, species, hybrids: instance size, see ``random_network``
    seed: seed of the simulation, so that a call is reproducible on its own

    Returns:
    False if some best match was not a weak best match, True otherwise.
    """
    seed_rngs(seed)
    heading(f"objective 1a: best matches vs weak best matches "
            f"({runs} networks, ~{leaves} leaves, {species} species, ~{hybrids} hybrids)")

    bm_is_wbm = wbm_is_bm = equal = 0
    counterexamples = []

    for _ in range(runs):
        network = random_network(leaves, species, hybrids)
        best = set(bmg_from_network(network).edges())
        weak = set(wbmg_from_network(network).edges())

        if best <= weak:
            bm_is_wbm += 1
        else:
            # the interesting case: a best match that is no weak best match
            counterexamples.append((network, best - weak))
        if weak <= best:
            wbm_is_bm += 1
        if best == weak:
            equal += 1

    print(result_line("best match => weak best match", bm_is_wbm, runs))
    print(result_line("weak best match => best match", wbm_is_bm, runs))
    print(result_line("BMG = WBMG", equal, runs))

    if counterexamples:
        network, missing = min(counterexamples, key=lambda c: c[0].number_of_nodes())
        print(f"\n    {len(counterexamples)} counterexample(s) to "
              f"'every best match is a weak best match'")
        print(f"    smallest one has {network.number_of_nodes()} vertices, "
              f"missing weak best matches: {sorted(map(str, missing))}")
        return False

    print("\n    no counterexample: every best match was also a weak best match")
    return True


def objective_1b(runs: int = 100, leaves: int = 10, species: int = 2,
                 hybrids: int = 3, seed: int = SEED) -> bool:
    """
    Round trip network -> WBMG -> network' -> WBMG' on random networks.

    Parameters:
    runs: number of networks to test
    leaves, species, hybrids: instance size, see ``random_network``
    seed: seed of the simulation, so that a call is reproducible on its own

    Returns:
    False if the construction returned a network with a different WBMG,
    True otherwise.
    """
    seed_rngs(seed)
    heading(f"objective 1b: BIC-cherry + extension for weak best matches "
            f"({runs} networks, ~{leaves} leaves, {species} species, ~{hybrids} hybrids)")

    explained = rejected = 0
    mismatches = []

    for _ in range(runs):
        wbmg = wbmg_from_network(random_network(leaves, species, hybrids))
        try:
            explanation = wbmg_cherry_extension(wbmg)
        except ValueError:
            # the construction refuses the graph, claiming no network explains it
            rejected += 1
            continue

        want, got = set(wbmg.edges()), set(wbmg_from_network(explanation).edges())
        if want == got:
            explained += 1
        else:
            mismatches.append((wbmg.number_of_nodes(), want - got, got - want))

    print(result_line("WBMG of the construction = WBMG", explained, runs))
    print(result_line("different WBMG", len(mismatches), runs))
    print(result_line("graph refused as no WBMG", rejected, runs))

    if mismatches:
        vertices, missing, extra = min(mismatches)
        print(f"\n    smallest mismatch has {vertices} leaves, {len(missing)} missing "
              f"and {len(extra)} spurious weak best matches")
        return False

    print("\n    the construction explained every weak best match graph it accepted")
    return True


def objective_1d(runs: int = 100, leaves: int = 10, species: int = 2,
                 hybrids: int = 3, seed: int = SEED) -> bool:
    """
    The round trip of 1b for the backtracking construction of ``bic_cherry_wmb``.

    Parameters:
    runs: number of networks to test
    leaves, species, hybrids: instance size, see ``random_network``
    seed: seed of the simulation, so that a call is reproducible on its own

    Returns:
    False if the construction returned a network with a different WBMG,
    True otherwise.
    """
    seed_rngs(seed)
    heading(f"objective 1d: BIC-cherry + expansion with backtracking "
            f"({runs} networks, ~{leaves} leaves, {species} species, ~{hybrids} hybrids)")

    explained = dead_ends = 0
    mismatches = []

    for _ in range(runs):
        wbmg = wbmg_from_network(random_network(leaves, species, hybrids))
        # a false edge that no extension partner removes is only warned about,
        # so the warning is captured instead of printed between the results
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            explanation = bic_cherry_for_wbm(wbmg)
        dead_ends += bool(warned)

        want, got = set(wbmg.edges()), set(wbmg_from_network(explanation).edges())
        if want == got:
            explained += 1
        else:
            mismatches.append((wbmg.number_of_nodes(), want - got, got - want))

    print(result_line("WBMG of the construction = WBMG", explained, runs))
    print(result_line("different WBMG", len(mismatches), runs))
    print(result_line("search ran into a dead end", dead_ends, runs))

    if mismatches:
        vertices, missing, extra = min(mismatches)
        print(f"\n    smallest mismatch has {vertices} leaves, {len(missing)} missing "
              f"and {len(extra)} spurious weak best matches")
        return False

    print("\n    the backtracking search explained every weak best match graph")
    return True


OBJECTIVES = {"1a": objective_1a, "1b": objective_1b, "1d": objective_1d}


def main(argv: list[str] | None = None) -> int:
    """Run the objectives named on the command line, or both of them."""
    wanted = list(argv if argv is not None else sys.argv[1:]) or list(OBJECTIVES)
    unknown = [name for name in wanted if name not in OBJECTIVES]
    if unknown:
        print(f"usage: main.py [{' | '.join(OBJECTIVES)}]  (unknown: {', '.join(unknown)})")
        return 2

    # every objective seeds itself, so running one alone gives the same
    # instances as running it as part of the whole script
    print(f"BMG project, objective 1 -- seed {SEED}")

    ok = True
    for name in wanted:
        ok &= OBJECTIVES[name]()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

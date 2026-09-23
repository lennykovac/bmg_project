from utils.graph_utils import *
from utils.tree_utils import create_gene_tree_n_leaves, import_good_trees
from utils.bic_cherry import bic_cherry_expansion, network_depth
from utils.report import table
from utils.lrt import (
    lrt_from_bmg,
    lrt_of_tree,
    is_bmg,
)

from utils.experiments import (
    arcs,
    compare_searches,
    complete_bmg,
    explains,
    minimize_bmg,
    pair_scan,
    single_move_scan,
    strict_search_fails,
)



'''
(a) Construct tree-BMGs (G, σ) from (AsymmeTree-generated) trees and
compute their least resolved trees. This is the target.
'''

leaves = 4
species = 2
species_tree_age = 1

tree = create_gene_tree_n_leaves(leaves, species, species_tree_age).gene_tree
# trees: list[nx.DiGraph] = import_good_trees()
tree_bmg = bmg_from_network(tree)
tree_lrt  = lrt_of_tree(tree)

leaves = leaves_from_network(tree)
for leaf in leaves:
    color = tree.nodes[leaf].get("color", "Keine Farbe")
    # print(f"Blatt {leaf!r} hat die Farbe/Spezies: {color}")

'''
(b) Take the tree-BMGs from (a) and construct the BIC-cherry+expansion
explanations (N, σ).
'''

N_bic_cherry = bic_cherry_expansion(tree_bmg, restricted=False)

'''
(c) Implement graph editing by “pulling up” or “pulling down” edges,
and removing vertices with the same parents and children, as de-
scribed in class.
'''

from utils.graph_editing import (
    pull_down,
    pull_up,
    merge_twins,  
)
from utils.graph_editing import *


'''
(d) Check both for single moves and combinations of a small number of
moves whether the modified network (N′, σ) has the same (weak) best match graph as (N, σ).
 
'''
 
# single_moves() applies ONE random move; combination() chains several.
# preserve_wbmg=True draws only among the moves that keep the WBMG, and
# combination() re-checks after EVERY step, so every intermediate network
# explains the same weak graph. Both raise NoLegalMove when nothing qualifies.
 
path_single: list = []
path_comb: list = []
path_guarded: list = []
 
# the reference is the WBMG of N itself -- NOT the tree's WBMG, which N does
# not explain in the first place (see (b))
wbmg_N = bmg_from_network(N_bic_cherry, weak=True)
 
N1_single = single_moves(N_bic_cherry, rng=0, record=path_single)
N1_comb = combination(N_bic_cherry, k=3, rng=0, record=path_comb)
try:
    N1_guarded = combination(
        N_bic_cherry, k=3, preserve_wbmg=True, rng=0, record=path_guarded)
    # True by construction -- if a move exists at all. On an expansion most
    # moves are safe anyway, so the guard often picks the same path as the
    # free draw; it earns its keep on the frozen instances of (f).
    guarded_line = (f"    combo   {' -> '.join(str(m) for m in path_guarded)}\n"
                    f"            WBMG kept: {explains(N1_guarded, wbmg_N, weak=True)}"
                    f"   (same path as the free draw: {path_guarded == path_comb})")
except NoLegalMove as exc:
    # not an edge case: for a complete bicolored BMG no move preserves the
    # graph at all, so the guarded editor cannot take a single step
    guarded_line = f"    no WBMG-preserving move exists -- {exc}"
 

print(f"""
(d) unguarded random edits
    single  {path_single[0]}
            WBMG kept: {explains(N1_single, wbmg_N, weak=True)}
    combo   {' -> '.join(str(m) for m in path_comb)}
            WBMG kept: {explains(N1_comb, wbmg_N, weak=True)}
 
    guarded (preserve_wbmg=True)
{guarded_line}""")
 
# One random draw says nothing about how *often* a move is safe. Every legal
# move, classified -- this is what task 2d actually asks.
scan = single_move_scan(N_bic_cherry, EXTENDED_MOVES)
totals = scan["ALL"]
 
# The interesting case is the one a per-step guard forbids: a first move that
# destroys the graph and a second that repairs it. Those pairs exist, which is
# why "every intermediate is valid" is strictly stronger than "the endpoints
# agree" -- see (f).
pairs = pair_scan(N_bic_cherry, EXTENDED_MOVES)
 
print(f"""
    every single move, classified
{table(("kind", "moves", "BMG kept", "WBMG kept", "both"),
       [(kind, row["total"], row["bmg"], row["wbmg"], row["both"])
        for kind, row in sorted(scan.items(), key=lambda kv: kv[0] == "ALL")])}
    -> {100 * totals['bmg'] / totals['total']:.0f}% keep the BMG, {100 * totals['wbmg'] / totals['total']:.0f}% keep the WBMG
 
    pairs of moves (exhaustive)
    {pairs['checked']} pairs, {pairs['kept']} keep the BMG, {pairs['rescued']} of those were
    'rescued' (first move broke it, second repaired it)""")
 
 
'''
(e) Devise heuristics to search for edit paths that allow the stepwise
conversion of the BIC-cherry+expansion explanations of (N, σ) in
the least resolved tree-explanation (lrt) T ∗ .
'''
 
from utils.search import RECOMMENDED
 
# The cost the descent minimises is a *guided* one: how far the cluster system
# of the current network is from that of T*, plus the reticulation number. A
# plain "is this cluster right" counter is flat almost everywhere and the
# search stalls after a handful of steps.
#
# strict=True      every intermediate network must explain G
# strict="prefer"  take an explaining step whenever one improves, and detour
#                  through a non-explaining network only when nothing else does
results = compare_searches(N_bic_cherry, tree_lrt)
best = results["extended/prefer"]
 
if best.reached_target:
    # depth is not an invariant of the moves: it is 3 after the expansion and
    # whatever T* needs at the end
    verdict = (
        f"    reached T* in {len(best.path)} moves; "
        f"same_phylogeny(result, T*) = {same_phylogeny(best.network, tree_lrt)}\n"
        f"    first moves: {' -> '.join(str(m) for m in best.path[:4])} ...\n"
        f"    depth {network_depth(N_bic_cherry)} -> {network_depth(best.network)} "
        f"(T* has {network_depth(tree_lrt)})")
else:
    verdict = (f"    did NOT reach T*: stopped at cost {best.cost}, "
               f"{best.network.number_of_nodes()} vertices")
 
print(f"""
(e) greedy descent from N towards T*
{table(("configuration", "steps", "-> T*", "tree", "explains", "cost"),
       [(label, len(r.path), r.reached_target, r.is_tree, r.explains, r.cost)
        for label, r in results.items()])}
 
    recommended: extended moves, strict={RECOMMENDED['strict']!r}, {RECOMMENDED['restarts']} restarts
{verdict}""")
 
 
'''
(f) If this is not always successful, find minimal examples that fail, try to
find out why they fail, and if possible use this information to improve
the heuristic for editing schedules.
'''
 
if strict_search_fails(tree_bmg, restricted=False):
    # shrink THIS instance: drop leaves one at a time while the failure lasts
    minimal = minimize_bmg(tree_bmg, lambda G: strict_search_fails(G, restricted=False))
    origin = (f"    this instance fails; minimal counterexample has "
              f"{minimal.number_of_nodes()} leaves (from {tree_bmg.number_of_nodes()})")
else:
    # it succeeded, so take the known minimal failure instead: the complete
    # bicolored BMG on three genes, whose LRT is the star
    minimal = complete_bmg((1, 2))
    origin = "    this instance succeeds -- inspecting the known minimal failure"
 
star = lrt_from_bmg(minimal)
N_min = reduce_twins(bic_cherry_expansion(minimal, restricted=False))
frozen = single_move_scan(N_min, EXTENDED_MOVES)["ALL"]
searches = compare_searches(N_min, star)
 
print(f"""
(f) failure analysis
{origin}
    {minimal.number_of_nodes()} leaves, {len({c for _, c in minimal.nodes(data='color')})} colors, {minimal.number_of_edges()} arcs
    T* is the star: {star.number_of_nodes() == minimal.number_of_nodes() + 1}
    expansion after twin merging: {N_min.number_of_nodes()} vertices, depth {network_depth(N_min)}
    (no extension: G has no missing arc)
    moves that preserve the BMG: {frozen['bmg']} of {frozen['total']}
    strict search:  {len(searches['named/strict'].path)} steps, reached T* = {searches['named/strict'].reached_target}
    relaxed search: {len(searches['extended/prefer'].path)} steps, reached T* = {searches['extended/prefer'].reached_target}
 
    Why it fails, and what fixes it
    -------------------------------
    For the complete bicolored BMG every gene is a best match of every gene of
    the other colour, so the expansion performs no extension and the network is
    the bare BIC-cherry network. Every lca then sits at the same level, and any
    single move lifts one gene above the others, which immediately changes the
    BMG. Zero preserving moves -- the strict search has no first step, even
    though the target (the star) is a perfectly ordinary tree.
 
    The obstruction is in the *move set*, not the construction, so a better
    expansion does not help. What helps is dropping the requirement that every
    intermediate network explain G: the relaxed rule reaches the target, and
    the 'rescued' pairs counted in (d) are the same phenomenon measured.""")
 


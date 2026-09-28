"""
Task 2.2 pipeline.

  (a) AsymmeTree species tree S, gene tree T, tree-BMG G = G(T, sigma);
      least resolved tree T* = Aho(R(G)) (Thm. 15), cross-checked against
      contraction of the redundant edges of T (Lemma 21) and AsymmeTree's
      lrt_from_tree.
  (b) BIC-cherry + expansion networks N (unrestricted and restricted).
  (c)-(d) guarded edits (utils.graph_editing).
  (e) search heuristics  N -> T*  (utils.network_search).
  (f) failures are stored (graph + final network) and the smallest ones printed.

usage:  python task2.py --instances 40 --max-leaves 8 --max-species 3 --seed 7
"""

import argparse
import csv
import json
import random
import time

import numpy as np
from asymmetree.analysis import lrt_from_tree, bmg_from_tree

from utils.bic_cherry import bic_cherry_expansion, restricted_bic_cherry_expansion
from utils.graph_utils import bmg_from_network, clusters, same_phylogeny, wbmg_from_network
from utils.lrt import is_least_resolved, lrt_by_contraction, lrt_from_bmg
from utils.network_search import edit_search, reduce_to_tree
from utils.tree_utils import create_gene_tree_n_leaves


def step_a(leaves, species):
    data = create_gene_tree_n_leaves(leaves, species)
    G = bmg_from_network(data.gene_tree)
    T_star = lrt_from_bmg(G)                                   # Thm. 15
    T_contr = lrt_by_contraction(data.gene_tree, G)            # Lemma 21 / Thm. 13
    # AsymmeTree's LRT may reuse leaf labels for inner vertices -> compare its
    # clusters directly on the tralda tree object
    T_asym = lrt_from_tree(data.original_gene_tree)
    below: dict = {}
    for v in T_asym.postorder():
        below[v] = frozenset([v.label]) if not v.children else frozenset().union(*(below[c] for c in v.children))
    asym_clusters = set(below.values())
    checks = {
        "aho==contraction": same_phylogeny(T_star, T_contr),
        "aho==asymmetree": set(clusters(T_star).values()) == asym_clusters,
        "least_resolved": is_least_resolved(T_star, G),
    }
    if not all(checks.values()):
        raise AssertionError(f"LRT cross-check failed: {checks}")
    return data, G, T_star


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, failures = [], []

    for i in range(args.instances):
        species = random.randint(2, args.max_species)
        leaves = random.randint(species, args.max_leaves)
        data, G, T_star = step_a(leaves, species)

        builders = {"bic": bic_cherry_expansion, "restricted": restricted_bic_cherry_expansion}
        for net_name, build in builders.items():
            N = build(G)                                  
            for mode in args.modes:
                compute = bmg_from_network if mode == "bmg" else wbmg_from_network
                explains = set(compute(N).edges) == set(G.edges)
                row = dict(
                    instance=i, leaves=G.number_of_nodes(), species=species,
                    network=net_name, mode=mode, N_nodes=N.number_of_nodes(),
                    N_edges=N.number_of_edges(), lrt_nodes=T_star.number_of_nodes(),
                    N_explains_G=explains,
                )
                if not explains:
                    # (weak) BMG of N differs from G -> T* is unreachable under this guard
                    row.update(greedy=False, agnostic=False, guided=False)
                    rows.append(row)
                    continue

                t = time.time()
                R, _ = reduce_to_tree(N, mode)
                row["greedy"] = same_phylogeny(R, T_star)
                for guided in (False, True):
                    key = "guided" if guided else "agnostic"
                    M, rep = edit_search(N, T_star, mode=mode, guided=guided, beam_width=args.beam)
                    if not rep.success and args.fallback_beam > args.beam:
                        M, rep = edit_search(N, T_star, mode=mode, guided=guided, beam_width=args.fallback_beam)
                        row[f"{key}_needed_fallback"] = rep.success

                    row[key] = rep.success
                    row[f"{key}_steps"] = rep.steps
                    row[f"{key}_guards"] = rep.evaluated
                    if not rep.success:
                        failures.append(dict(
                            instance=i, network=net_name, mode=mode, search=key,
                            leaves=G.number_of_nodes(), species=species,
                            colors={str(v): c for v, c in G.nodes(data="color")},
                            bmg_edges=[list(map(str, e)) for e in G.edges],
                            lrt_clusters=sorted(sorted(map(str, c)) for v, c in clusters(T_star).items()
                                                if T_star.out_degree(v)),
                            final_edges=[[sorted(map(str, clusters(M)[u])), sorted(map(str, clusters(M)[v]))]
                                         for u, v in M.edges],
                            final_is_tree=rep.final_is_tree, last_moves=[str(m) for m in rep.path[-6:]],
                        ))
                row["seconds"] = round(time.time() - t, 2)
                rows.append(row)
        print(f"[{i + 1}/{args.instances}] |L|={G.number_of_nodes()} |S|={species} "
              + " ".join(f"{r['network']}/{r['mode']}: g={r['greedy']:d} a={r['agnostic']:d} t*={r['guided']:d}"
                         for r in rows if r["instance"] == i), flush=True)

    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    failures.sort(key=lambda f: (f["leaves"], f["species"]))
    with open(args.out + "_failures.json", "w") as fh:
        json.dump(failures, fh, indent=1)

    print("\n=== summary: success rate  N -> T* ===")
    for net in ("bic", "restricted"):
        for mode in args.modes:
            sel = [r for r in rows if r["network"] == net and r["mode"] == mode]
            if not sel:
                continue
            n = len(sel)
            expl = sum(r["N_explains_G"] for r in sel)
            print(f"{net:10s} {mode:4s}  N explains G: {expl}/{n}   "
                  + "  ".join(f"{k}: {sum(r[k] for r in sel)}/{n}" for k in ("greedy", "agnostic", "guided"))
                  + "   (solved only by fallback beam: "
                  + ", ".join(f"{k} {sum(bool(r.get(k + '_needed_fallback')) for r in sel)}" for k in ("agnostic", "guided"))
                  + ")")
    if failures:
        f = failures[0]
        print(f"\nsmallest failure ({f['search']}, {f['network']}/{f['mode']}): |L|={f['leaves']}")
        print(" colors:", f["colors"])
        print(" BMG:", f["bmg_edges"])
        print(" T* clusters:", f["lrt_clusters"])
        print(" final network (cluster -> cluster):", f["final_edges"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--max-leaves", type=int, default=8)
    ap.add_argument("--max-species", type=int, default=4)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--fallback-beam", type=int, default=3, help="retry failures with this beam width (0 = off)")
    ap.add_argument("--modes", nargs="+", default=["bmg", "wbmg"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results_task2")
    run(ap.parse_args())

"""Task 2e/2f over many AsymmeTree instances, in parallel.

    python batch_task_2.py --n 60 --min-leaves 4 --max-leaves 12 --species 4 --workers 8

For every instance: BMG G, target T*, restricted BIC-cherry expansion N, then
every configuration of CONFIGS plus two ablations of the guidance
(which move ordering / pruning rule is responsible for what). For instances
of at most --proof-leaves genes, the atomic strict search is additionally run
exhaustively (proven_failure) and proven failures are shrunk to minimal
counterexamples, which are grouped by isomorphism class.

All searches are independent, so they are distributed over processes
(ProcessPoolExecutor); results go to --out as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.edit_search import ATOMIC, CONFIGS, WITH_MACROS, _pool_context, solve_many
from task_2_task_2_utils.experiments import generate_instances
from task_2_utils.failures import minimal_counterexample, proven_failure, signature
from utils.graph_utils import bmg_from_network
from utils.lrt import lrt_from_bmg
from task_2_utils.report import table

NO_GROUPS = tuple(k for k in WITH_MACROS if k != "merge_group")

ALL_CONFIGS = {
    **CONFIGS,
    # ablations of the heuristic (macro moves, strict invariant)
    "macro/strict -groups": dict(kinds=NO_GROUPS, mode="strict"),
    "macro/strict -groups -prune": dict(kinds=NO_GROUPS, mode="strict",
                                        prune=False, fallback=False),
    # beam search instead of first-improvement (same moves, same invariant)
    "macro/strict beam": dict(kinds=WITH_MACROS, mode="strict", strategy="beam",
                              width=8, per_state=30),
    "macro/strict beam A*": dict(kinds=WITH_MACROS, mode="strict", strategy="beam",
                                 width=16, per_state=50, beam_weight=1.0),
}


def _proof(args):
    i, G = args
    verdict = proven_failure(G, kinds=ATOMIC)
    sig = signature(minimal_counterexample(G, kinds=ATOMIC)) if verdict else None
    return i, verdict, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-leaves", type=int, default=4)
    ap.add_argument("--max-leaves", type=int, default=10)
    ap.add_argument("--species", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--proof-leaves", type=int, default=7)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--out", default="results_task_2.json")
    a = ap.parse_args()
    t0 = time.time()

    trees = generate_instances(a.n, min_leaves=a.min_leaves, max_leaves=a.max_leaves,
                               max_species=a.species, seed=a.seed)
    inst = []
    for t in trees:
        G = bmg_from_network(t)
        inst.append((G, lrt_from_bmg(G), bic_cherry_expansion(G, restricted=True)))

    labels = a.configs.split(",")
    jobs = [((i, lab), N, T, dict(ALL_CONFIGS[lab], budget=a.budget, verify=True))
            for i, (G, T, N) in enumerate(inst) for lab in labels]
    results = solve_many(jobs, workers=a.workers)

    rows, record = [], {lab: [] for lab in labels}
    for (i, lab), r in results:
        record[lab].append(dict(instance=i, leaves=inst[i][0].number_of_nodes(),
                                n_vertices=inst[i][2].number_of_nodes(),
                                reached=r.reached, atomic_moves=len(r.path),
                                search_steps=r.macro_steps, broken_steps=r.broken_steps,
                                broken_atomic=r.broken_atomic, verified=r.verified,
                                expanded=r.expanded, seconds=r.seconds))
    for lab in labels:
        rs = record[lab]
        ok = [r for r in rs if r["reached"]]
        rows.append((lab, f"{len(ok)}/{len(rs)}",
                     f"{sum(r['atomic_moves'] for r in ok) / max(1, len(ok)):.1f}",
                     f"{sum(r['broken_atomic'] or 0 for r in ok) / max(1, len(ok)):.1f}",
                     all(r["verified"] for r in ok),
                     f"{sum(r['seconds'] for r in rs):.1f}s"))
    print(f"\n{len(inst)} instances, {a.min_leaves}-{a.max_leaves} genes, "
          f"<= {a.species} species, restricted BIC-cherry expansion")
    print(table(("configuration", "-> T*", "avg moves", "avg broken atomic",
                 "all replays ok", "cpu time"), rows))

    # -- 2f: proofs and minimal counterexamples -------------------------------
    small = [(i, G) for i, (G, _T, _N) in enumerate(inst)
             if G.number_of_nodes() <= a.proof_leaves]
    workers = max(1, a.workers or 1)
    if workers > 1 and len(small) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
            proofs = list(pool.map(_proof, small))
    else:
        proofs = [_proof(x) for x in small]
    verdicts = Counter({True: "no path (proof)", False: "path exists",
                        None: "undecided"}[v] for _i, v, _s in proofs)
    sigs = Counter(s for _i, v, s in proofs if v)
    print(f"\n2f  atomic/strict, exhaustive, instances with <= {a.proof_leaves} genes:"
          f" {dict(verdicts)}")
    for s, k in sigs.most_common():
        print(f"    minimal counterexample {s}   ({k}x)")

    json.dump(dict(args=vars(a), searches=record,
                   proofs=[dict(instance=i, verdict=v, minimal=s) for i, v, s in proofs]),
              open(a.out, "w"), indent=1)
    print(f"\nwritten to {a.out}   ({time.time() - t0:.0f}s wall, {a.workers} workers)")


if __name__ == "__main__":
    main()

"""first-improvement vs beam on the same instances (macro moves, strict)."""
import sys, time
from task_2_task_2_utils.experiments import generate_instances
from utils.graph_utils import bmg_from_network
from utils.lrt import lrt_from_bmg
from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.edit_search import find_path, WITH_MACROS
from task_2_utils.report import table
V = {"first":            dict(strategy="first"),
     "beam w=4  k=10":   dict(strategy="beam", width=4, per_state=10),
     "beam w=8  k=30":   dict(strategy="beam", width=8, per_state=30),
     "beam w=16 k=50":   dict(strategy="beam", width=16, per_state=50),
     "beam A* w=1":      dict(strategy="beam", width=16, per_state=50, beam_weight=1.0),
     "beam A* w=2":      dict(strategy="beam", width=16, per_state=50, beam_weight=2.0),
     "beam A* w=4":      dict(strategy="beam", width=16, per_state=50, beam_weight=4.0)}
n, lo, hi, sp, seed = map(int, sys.argv[1:6])
names = sys.argv[6].split(",") if len(sys.argv) > 6 else list(V)
inst = []
for t in generate_instances(n, min_leaves=lo, max_leaves=hi, max_species=sp, seed=seed):
    G = bmg_from_network(t); inst.append((lrt_from_bmg(G), bic_cherry_expansion(G, restricted=True)))
rows = []; per = {v: [] for v in names}
for T, N in inst:
    for v in names:
        r = find_path(N, T, kinds=WITH_MACROS, mode="strict", budget=60000, verify=True, **V[v])
        per[v].append(r)
for v in names:
    rs = per[v]; ok = [r for r in rs if r.reached]
    rows.append((v, f"{len(ok)}/{len(rs)}", f"{sum(len(r.path) for r in ok)/max(1,len(ok)):.1f}",
                 f"{sum(r.macro_steps for r in ok)/max(1,len(ok)):.1f}",
                 f"{sum(r.broken_atomic for r in ok)/max(1,len(ok)):.1f}",
                 all(r.verified for r in ok), f"{sum(r.expanded for r in rs)}",
                 f"{sum(r.seconds for r in rs):.1f}s"))
# paired comparison of path length against 'first' on instances both solved
if "first" in names:
    for v in names[1:]:
        pairs = [(a, b) for a, b in zip(per["first"], per[v]) if a.reached and b.reached]
        shorter = sum(len(b.path) < len(a.path) for a, b in pairs)
        longer = sum(len(b.path) > len(a.path) for a, b in pairs)
        d = [len(b.path) - len(a.path) for a, b in pairs]
        print(f"  {v}: shorter than first on {shorter}, longer on {longer}, of {len(pairs)};"
              f" paired mean difference {sum(d)/max(1,len(d)):+.1f} atomic moves"
              f" (min {min(d, default=0)}, max {max(d, default=0)})")
print(table(("strategy", "-> T*", "avg atomic", "avg steps", "avg broken atomic",
             "replay ok", "candidates", "time"), rows))

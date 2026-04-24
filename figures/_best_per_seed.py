"""Print top-3 (K,T) configurations per seed with AUC, H_s, H_p."""
import csv
from pathlib import Path

CSV = Path(__file__).parent / "mk_grid_summary.csv"
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append(dict(
            seed=int(r["seed"]), K=int(r["K"]), T=int(r["T"]),
            AUC=float(r["AUC"]), H_s=float(r["H_s"]), H_p=float(r["H_p"]),
        ))

seeds = sorted(set(r["seed"] for r in rows))
print(f"{'seed':>5}  {'rank':>4}  {'K':>3} {'T':>3}  {'AUC':>7}  {'H_s':>6}  {'H_p':>6}")
print("-" * 50)
for s in seeds:
    sr = sorted([r for r in rows if r["seed"] == s], key=lambda x: x["AUC"], reverse=True)
    for rank, r in enumerate(sr[:3], 1):
        print(f"{s:>5}  {rank:>4}  {r['K']:>3} {r['T']:>3}  {r['AUC']:>7.4f}  {r['H_s']:>6.3f}  {r['H_p']:>6.3f}")
    print()

# Mean over seeds at each (K, T)
from collections import defaultdict
import statistics as stat
agg = defaultdict(list)
for r in rows:
    agg[(r["K"], r["T"])].append(r["AUC"])
mean_sorted = sorted([(k, t, stat.mean(v), stat.stdev(v) if len(v) > 1 else 0.0)
                      for (k, t), v in agg.items()], key=lambda x: x[2], reverse=True)
print("=" * 50)
print("Mean over 5 seeds — top-5 (K,T)")
print(f"{'rank':>4}  {'K':>3} {'T':>3}  {'mean AUC':>9}  {'std':>6}")
for i, (k, t, m, sd) in enumerate(mean_sorted[:5], 1):
    print(f"{i:>4}  {k:>3} {t:>3}  {m:>9.4f}  {sd:>6.4f}")

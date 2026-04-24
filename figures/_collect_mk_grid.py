"""
Collect per-seed (K, T) grid results from the mk_grid sweep.

For each config directory under the sweep root:
  - Parse the JSON to get AUC + run_tag + hyperparameters (K=n_graphs, T=n_nodes, seed)
  - Use run_tag to locate the checkpoint training log
  - Parse the last [ROUTING] line in the log to extract H_s and H_p

Writes a CSV summary to figures/mk_grid_summary.csv
"""

import os
import re
import json
import glob
import csv
import sys
from pathlib import Path

SWEEP_ROOT = Path(
    "/storage/personal/eungyeop/experiments/experiments/"
    "source_to_source_sweep_mk_grid_20260418/"
    "Medicaldataset+Cardiovascular_Disease_Dataset+Heart_disease_statlog+"
    "Erbil_Cardiovascular_Health_Dataset+cardio_SAheart+heart_failure_clinical_records"
)
CHECKPOINT_ROOT = Path(
    "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/"
    "Medicaldataset+Cardiovascular_Disease_Dataset+Heart_disease_statlog+"
    "Erbil_Cardiovascular_Health_Dataset+cardio_SAheart+heart_failure_clinical_records/Pre"
)

ROUTING_RE = re.compile(
    r"\[ROUTING\].*?H_s=([0-9.]+)\s+H_p=([0-9.]+)"
)

# also capture the top1 just for diagnostics
TOP1_RE = re.compile(r"top1=([0-9.]+)")


def find_log_for_run_tag(run_tag: str, seed: int):
    """Find train_log_<run_tag>.log under CHECKPOINT_ROOT/**/<seed>/<run_tag>/"""
    pattern = str(CHECKPOINT_ROOT / "**" / str(seed) / run_tag / f"train_log_{run_tag}.log")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None
    # Usually only one
    return hits[0]


def parse_last_routing(log_path: str, tail_n: int = 20):
    """Parse the last `tail_n` [ROUTING] lines and return mean H_s, H_p.

    Averaging the tail instead of taking strictly the last line gives a
    more stable estimate (the values fluctuate from batch to batch).
    """
    hs_vals = []
    hp_vals = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "[ROUTING]" not in line:
                continue
            m = ROUTING_RE.search(line)
            if m:
                hs_vals.append(float(m.group(1)))
                hp_vals.append(float(m.group(2)))
    if not hs_vals:
        return None, None, 0
    tail_hs = hs_vals[-tail_n:]
    tail_hp = hp_vals[-tail_n:]
    return (sum(tail_hs) / len(tail_hs), sum(tail_hp) / len(tail_hp), len(hs_vals))


def main():
    rows = []
    seed_dirs = sorted([p for p in SWEEP_ROOT.iterdir() if p.name.startswith("args_seed:")])
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split(":")[1])
        config_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
        for cd in config_dirs:
            json_files = list(cd.glob("*.json"))
            if not json_files:
                print(f"[WARN] no json in {cd}", file=sys.stderr)
                continue
            # take the most recent
            jf = sorted(json_files)[-1]
            with open(jf) as f:
                data = json.load(f)
            hp = data["hyperparameters"]
            K = int(hp["n_graphs"])
            T = int(hp["n_nodes"])
            run_tag = hp["run_tag"]
            auc = None
            try:
                auc = float(data["results"]["Ours_few"]["Ours_best_few_auc"])
            except Exception as e:
                print(f"[WARN] no AUC in {jf}: {e}", file=sys.stderr)

            log_path = find_log_for_run_tag(run_tag, seed)
            Hs = Hp = None
            nrout = 0
            if log_path is None:
                print(f"[WARN] log not found for run_tag={run_tag} seed={seed}", file=sys.stderr)
            else:
                Hs, Hp, nrout = parse_last_routing(log_path, tail_n=20)

            rows.append(dict(seed=seed, K=K, T=T, AUC=auc, H_s=Hs, H_p=Hp,
                             n_routing_steps=nrout, run_tag=run_tag,
                             log_path=log_path or ""))

    out = Path(__file__).parent / "mk_grid_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")

    # quick per-seed summary print
    seeds = sorted(set(r["seed"] for r in rows))
    print("\n=== per-seed completeness ===")
    print(f"{'seed':>5} {'n_rows':>7} {'with_AUC':>9} {'with_Hs':>8}")
    for s in seeds:
        srows = [r for r in rows if r["seed"] == s]
        n_auc = sum(1 for r in srows if r["AUC"] is not None)
        n_hs = sum(1 for r in srows if r["H_s"] is not None)
        print(f"{s:>5} {len(srows):>7} {n_auc:>9} {n_hs:>8}")


if __name__ == "__main__":
    main()

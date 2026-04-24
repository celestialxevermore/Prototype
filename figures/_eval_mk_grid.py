"""
Recompute H_s / H_p for every run in the mk_grid sweep by:
  - loading best_joint.pt
  - rebuilding the model from ckpt['args']
  - running forward over ALL source-data samples in eval mode
  - aggregating pi from model.graph_quantizer.last_pi
  - computing H_s (mean per-sample entropy / log K) and H_p (marginal entropy / log K)

Writes to mk_grid_summary_eval.csv (does NOT touch mk_grid_summary.csv).
"""
import os, sys, json, math, csv, glob, time, traceback
from pathlib import Path
import torch

ROOT = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217"
sys.path.insert(0, ROOT)

from models.TabularFLM_S_ import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders

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
OUT_CSV = Path(ROOT) / "figures" / "mk_grid_summary_eval.csv"


def find_ckpt(run_tag: str, seed: int):
    pattern = str(CHECKPOINT_ROOT / "**" / str(seed) / run_tag / "best_joint.pt")
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def compute_entropies(model, source_data, args, device):
    """Forward over all source samples, return (H_s, H_p, N, K)."""
    pis = []
    with torch.no_grad():
        for src in source_data:
            res = prepare_embedding_dataloaders(args, src, is_source=True)
            train_loader = res['loaders'][0]
            for batch in train_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                _ = model.predict(batch, return_all=True)
                pi = model.graph_quantizer.last_pi
                pis.append(pi.detach().cpu())

    pi_all = torch.cat(pis, dim=0)
    N, K = pi_all.shape
    pi_safe = pi_all.clamp_min(1e-12)
    Hmax = math.log(K)
    H_s = float((-(pi_safe * pi_safe.log()).sum(dim=1)).mean().item() / Hmax)
    p = pi_all.mean(dim=0)
    p_safe = p.clamp_min(1e-12)
    H_p = float((-(p_safe * p_safe.log()).sum()).item() / Hmax)
    p_max = float(p.max().item())           # how concentrated is the most-used graph
    return H_s, H_p, N, K, p_max


def iter_runs():
    """Yield (seed, K, T, run_tag, AUC, json_path)."""
    for seed_dir in sorted(SWEEP_ROOT.iterdir()):
        if not seed_dir.name.startswith("args_seed:"):
            continue
        seed = int(seed_dir.name.split(":")[1])
        for cd in sorted(seed_dir.iterdir()):
            jfs = sorted(cd.glob("*.json"))
            if not jfs:
                continue
            with open(jfs[-1]) as f:
                data = json.load(f)
            hp = data["hyperparameters"]
            K = int(hp["n_graphs"]); T = int(hp["n_nodes"])
            rt = hp["run_tag"]
            try:
                auc = float(data["results"]["Ours_few"]["Ours_best_few_auc"])
            except Exception:
                auc = None
            yield seed, K, T, rt, auc, str(jfs[-1])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    runs = list(iter_runs())
    print(f"total runs: {len(runs)}")

    rows = []
    t0 = time.time()
    for idx, (seed, K, T, run_tag, auc, json_path) in enumerate(runs, 1):
        ckpt_path = find_ckpt(run_tag, seed)
        if ckpt_path is None:
            print(f"[{idx}/{len(runs)}] MISSING ckpt seed={seed} K={K} T={T} run_tag={run_tag}")
            rows.append(dict(seed=seed, K=K, T=T, AUC=auc, H_s=None, H_p=None,
                             p_max=None, N=0, run_tag=run_tag, ckpt_path="",
                             status="missing_ckpt"))
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            args = ckpt['args']
            args.use_lcg = True

            model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                          args.dropout_rate, args.llm_model,
                          experiment_id="eval", mode="Full").to(device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            model.eval()

            H_s, H_p, N, K_ckpt, p_max = compute_entropies(
                model, args.source_data, args, device
            )
            assert K_ckpt == K, f"K mismatch: ckpt={K_ckpt} vs json={K}"

            del model
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            eta = elapsed / idx * (len(runs) - idx)
            print(f"[{idx}/{len(runs)}] seed={seed} K={K:>2} T={T:>2}  "
                  f"H_s={H_s:.4f} H_p={H_p:.4f} p_max={p_max:.3f} N={N}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            rows.append(dict(seed=seed, K=K, T=T, AUC=auc, H_s=H_s, H_p=H_p,
                             p_max=p_max, N=N, run_tag=run_tag, ckpt_path=ckpt_path,
                             status="ok"))
        except Exception as e:
            print(f"[{idx}/{len(runs)}] FAILED seed={seed} K={K} T={T}: {e}")
            traceback.print_exc()
            rows.append(dict(seed=seed, K=K, T=T, AUC=auc, H_s=None, H_p=None,
                             p_max=None, N=0, run_tag=run_tag, ckpt_path=ckpt_path,
                             status=f"error:{type(e).__name__}"))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()

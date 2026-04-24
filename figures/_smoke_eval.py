"""
Smoke test: reload ONE ckpt, compute H_s / H_p over source data, compare to tail-20 value.
"""
import os, sys, math
import torch

# Path so we can import project modules
ROOT = "/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217"
sys.path.insert(0, ROOT)

from models.TabularFLM_S_ import Model
from dataset.data_dataloaders import prepare_embedding_dataloaders

CKPT_PATH = (
    "/storage/personal/eungyeop/experiments/checkpoints/gpt2_mean/"
    "Medicaldataset+Cardiovascular_Disease_Dataset+Heart_disease_statlog+"
    "Erbil_Cardiovascular_Health_Dataset+cardio_SAheart+heart_failure_clinical_records/Pre/"
    "ngraphs-8_nnodes-12_gdim-768_nbasis-2_basis-ind_attn-gat_v1_struct_hidden_dim-192_"
    "fgw_alpha-1.5_alpha-0.5_vq_beta-0.3_kl_gamma-2.0_tau-0.5_target_data-heart_"
    "entropic_reg-0.01_description-None/42/20260419_000808/best_joint.pt"
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    args = ckpt['args']
    print(f"args.n_graphs={args.n_graphs} args.n_nodes={args.n_nodes} seed={args.random_seed}")
    print(f"source_data: {args.source_data}")

    # Force the LCG branch on
    args.use_lcg = True

    model = Model(args, args.input_dim, args.hidden_dim, args.output_dim,
                  args.dropout_rate, args.llm_model, experiment_id="smoke", mode="Full").to(device)
    msg = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"load msg: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    model.eval()

    pis = []
    with torch.no_grad():
        for src in args.source_data:
            res = prepare_embedding_dataloaders(args, src, is_source=True)
            train_loader = res['loaders'][0]
            n_batches = 0
            for batch in train_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                _ = model.predict(batch, return_all=True)
                pi = model.graph_quantizer.last_pi  # [B, K]
                pis.append(pi.detach().cpu())
                n_batches += 1
            print(f"  src={src}: {n_batches} batches")

    pi_all = torch.cat(pis, dim=0)           # [N_total, K]
    N, K = pi_all.shape
    print(f"\npi_all shape: [{N}, {K}]")

    pi_safe = pi_all.clamp_min(1e-12)
    H_max = math.log(K)

    # Per-sample entropy averaged over samples
    H_s = (-(pi_safe * pi_safe.log()).sum(dim=1)).mean().item() / H_max

    # Marginal entropy
    p = pi_all.mean(dim=0)                    # [K]
    p_safe = p.clamp_min(1e-12)
    H_p = (-(p_safe * p_safe.log()).sum()).item() / H_max

    print(f"\n=== SMOKE RESULT (source-pool, all samples, eval mode) ===")
    print(f"  H_s (per-sample)     = {H_s:.4f}")
    print(f"  H_p (batch-marginal) = {H_p:.4f}")
    print(f"  effective K          = {K * math.exp((H_p - 1) * H_max):.2f} / {K}")

    print(f"\n[reference] tail-20 training log values for this run (from mk_grid_summary.csv):")
    print(f"  H_s ≈ 0.150, H_p ≈ 0.852  (noisy, training-time)")


if __name__ == "__main__":
    main()

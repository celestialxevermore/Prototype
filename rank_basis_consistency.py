"""
Side-rank c1-perfect candidates by PER-BASIS semantic consistency across cluster1 datasets.

기존 c1_semantic (top-2 Jaccard on common basis) 대비 개선점:
  1. c1 union 의 모든 basis 에 대해 평가 (intersection 아님)
  2. profile magnitude 를 semantic group 으로 aggregate 한 vector 간 cosine → 연속값
  3. basis 가 몇 개 ds 에서 나타나는지 (n_ds)^2 로 weight (3 ds 전부 활성화되면 4x)

사용:
  python rank_basis_consistency.py \
      --scores /tmp/spec_scores_p1.csv /tmp/spec_scores_p2.csv \
      --pre <P1_Pre_dir> <P2_Pre_dir> \
      --out /tmp/basis_consistency.csv --device cuda:4
"""
import argparse, os, sys, glob, traceback
import numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path('/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217')
sys.path.append(str(ROOT))

from specialization_search import (
    CLUSTER1, FEATURE_GROUPS, feature_to_group,
    compute_basis_profile, pick_active_basis,
)
from analysis2 import (
    load_model, load_source_loaders,
    compute_onehot_basis_risk, extract_wpr_with_tabular,
)


def basis_group_vector(profile, num_cols, pos_only=True):
    groups = list(FEATURE_GROUPS.keys()) + ['OTHER']
    g2i = {g: i for i, g in enumerate(groups)}
    vec = np.zeros(len(groups))
    prof = np.asarray(profile, dtype=float)
    if pos_only:
        prof = np.clip(prof, 0, None)
    else:
        prof = np.abs(prof)
    for f, v in zip(num_cols, prof):
        vec[g2i[feature_to_group(f)]] += v
    return vec


def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-9))


def per_basis_consistency(dataset_info, c1_srcs):
    c1_union_h = set()
    for s in c1_srcs:
        c1_union_h.update(dataset_info[s]['h_basis'])
    if not c1_union_h:
        return 0.0, {}

    per_basis = {}
    w_sum, w_tot = 0.0, 0
    for k in c1_union_h:
        vecs, ds_labels = [], []
        for s in c1_srcs:
            prof_map = dataset_info[s]['profiles']
            cols = dataset_info[s]['num_cols']
            if k in prof_map and cols is not None:
                v = basis_group_vector(prof_map[k]['profile'], cols)
                if np.linalg.norm(v) > 1e-6:
                    vecs.append(v); ds_labels.append(s)
        if len(vecs) < 2:
            per_basis[k] = {'score': None, 'n_ds': len(vecs)}
            continue
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(cosine(vecs[i], vecs[j]))
        score = float(np.mean(sims))
        per_basis[k] = {'score': score, 'n_ds': len(vecs)}
        w = len(vecs) ** 2
        w_sum += score * w; w_tot += w
    return (w_sum / w_tot if w_tot else 0.0), per_basis


def peak_groups_per_basis(dataset_info, c1_srcs, top_n=2):
    rows = {}
    c1_union = set()
    for s in c1_srcs: c1_union.update(dataset_info[s]['h_basis'])
    for k in sorted(c1_union):
        entries = []
        for s in c1_srcs:
            prof_map = dataset_info[s]['profiles']
            cols = dataset_info[s]['num_cols']
            if k in prof_map and cols is not None:
                vec = basis_group_vector(prof_map[k]['profile'], cols)
                groups = list(FEATURE_GROUPS.keys()) + ['OTHER']
                idx = np.argsort(-vec)[:top_n]
                pg = '+'.join(groups[i] for i in idx if vec[i] > 1e-3)
                entries.append(f'{s[:6]}:{pg}')
            else:
                entries.append(f'{s[:6]}:-')
        rows[f'B{k}'] = ' | '.join(entries)
    return rows


def score_checkpoint(ckpt_path, device):
    model, model_args = load_model(ckpt_path, device)
    sources, loaders = load_source_loaders(model_args)
    onehot_probs = compute_onehot_basis_risk(model, device)
    all_results = {}
    for src in sources:
        try:
            all_results[src] = extract_wpr_with_tabular(model, loaders[src], device, src)
        except Exception:
            pass
    dataset_info = {}
    for src in sources:
        if src not in all_results: continue
        profiles, cols = compute_basis_profile(all_results[src], onehot_probs)
        if not profiles: continue
        h_basis, l_ref = pick_active_basis(profiles)
        dataset_info[src] = {'h_basis': h_basis, 'l_ref': l_ref,
                             'profiles': profiles, 'num_cols': cols}
    c1_srcs = [s for s in CLUSTER1 if s in dataset_info]
    if len(c1_srcs) < 2:
        return None
    consistency, per_basis = per_basis_consistency(dataset_info, c1_srcs)
    peaks = peak_groups_per_basis(dataset_info, c1_srcs)
    return consistency, per_basis, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--pre', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:4')
    args = ap.parse_args()
    device = torch.device(args.device)

    rows_out = []
    if os.path.exists(args.out):
        rows_out = pd.read_csv(args.out).to_dict('records')
        done_tags = {r['run_tag'] for r in rows_out}
        print(f'Resume: {len(done_tags)} already scored')
    else:
        done_tags = set()

    for csv_path, pre_dir in zip(args.scores, args.pre):
        df = pd.read_csv(csv_path)
        ok = df[(df['c1_jaccard']==1.0) & (df['c1_lref_agree']==1)]
        print(f'[{csv_path}] total={len(df)}  c1-perfect={len(ok)}')
        ok_tags = set(ok['run_tag'].astype(str).tolist())

        ckpt_names = ['best_joint.pt', 'best_val_joint.pt', 'best.pt', 'latest.pt']
        found = []
        for name in ckpt_names:
            found.extend(glob.glob(os.path.join(pre_dir, '**', name), recursive=True))
        by_dir = {}
        for p in found:
            d = os.path.dirname(p)
            if d not in by_dir or os.path.basename(p) == 'best_joint.pt':
                by_dir[d] = p
        ckpts = [p for p in sorted(by_dir.values())
                 if os.path.basename(os.path.dirname(p)) in ok_tags
                 and os.path.basename(os.path.dirname(p)) not in done_tags]
        print(f'  to-score: {len(ckpts)}')

        for i, ckpt in enumerate(ckpts, 1):
            run_tag = os.path.basename(os.path.dirname(ckpt))
            try:
                res = score_checkpoint(ckpt, device)
                if res is None:
                    continue
                consistency, per_basis, peaks = res
                row = {'run_tag': run_tag, 'csv': os.path.basename(csv_path),
                       'consistency': consistency}
                row.update({f'peaks_{k}': v for k, v in peaks.items()})
                row.update({f'score_{k}': (v['score'] if v['score'] is not None else '')
                            for k, v in per_basis.items()})
                row.update({f'nds_{k}': v['n_ds'] for k, v in per_basis.items()})
                rows_out.append(row)
                if i % 3 == 0 or i == len(ckpts):
                    print(f'  [{i}/{len(ckpts)}] {run_tag} consistency={consistency:.3f}')
                    pd.DataFrame(rows_out).to_csv(args.out, index=False)
            except Exception as e:
                print(f'  [err] {run_tag}: {type(e).__name__}: {e}')
                traceback.print_exc()

    pd.DataFrame(rows_out).to_csv(args.out, index=False)
    print(f'\nSaved {len(rows_out)} rows → {args.out}')


if __name__ == '__main__':
    main()

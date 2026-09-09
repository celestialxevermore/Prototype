"""final_grid_20260908 집계: config x seed -> per-seed 표 + mean/std."""
import json, glob, re, os, csv
from pathlib import Path
import numpy as np

EXP = Path("/storage/personal/eungyeop/experiments/experiments")
LOGDIR = Path("/storage/personal/eungyeop/dataset/logs/final_grid_20260908")
OUT = Path("/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures")

ROUTING = re.compile(r"\[ROUTING\].*?H_s=([0-9.]+)\s+H_p=([0-9.]+).*?top1=([0-9.]+)")
COST    = re.compile(r"Cost Scale \| Feat: ([0-9.]+) vs Struct: ([0-9.]+) \| Ratio: ([0-9.]+)")

rows = []
for ds in ("breast", "cvd"):
    for tag in ("cos_st05", "cos_st01", "l2_st05", "l2_st01"):
        d = EXP / f"source_to_source_{ds}_case1_{tag}_final_20260908" / "case1"
        for seed in (42, 44, 46, 48, 50):
            hits = sorted(glob.glob(str(d / f"*seed{seed}_*.json")))
            log = LOGDIR / f"{ds}_{tag}_seed{seed}.log"
            r = dict(dataset=ds, config=tag, seed=seed,
                     feat_distance="cosine" if tag.startswith("cos") else "l2",
                     soft_tau=0.5 if tag.endswith("st05") else 0.1)
            if hits:
                j = json.load(open(hits[-1]))
                pst = j.get("per_source_test", {})
                for k in ("auc", "auprc", "acc", "f1"):
                    v = pst.get(k)
                    if v: r[f"mean_{k}"] = float(np.mean(v))
                r["sources"] = "+".join(pst.get("sources", []))
                r["status"] = "ok"
            else:
                r["status"] = "missing"
            if log.exists():
                txt = log.read_text(errors="ignore")
                rt = ROUTING.findall(txt)
                if rt:
                    r["H_s"], r["H_p"], r["top1"] = (float(rt[-1][0]), float(rt[-1][1]), float(rt[-1][2]))
                cs = COST.findall(txt)
                if cs:
                    r["feat_term"], r["struct_term"], r["ratio"] = (float(cs[-1][0]), float(cs[-1][1]), float(cs[-1][2]))
                m = re.search(r"l2 feat_scale\] q90=([0-9.]+)", txt)
                if m: r["feat_scale_q90"] = float(m.group(1))
            rows.append(r)

cols = ["dataset","config","feat_distance","soft_tau","seed","status",
        "mean_auc","mean_auprc","mean_acc","mean_f1",
        "H_s","H_p","top1","feat_term","struct_term","ratio","feat_scale_q90"]
with open(OUT/"final_grid_per_seed.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)

# mean/std
agg = []
for ds in ("breast","cvd"):
    for tag in ("cos_st05","cos_st01","l2_st05","l2_st01"):
        sub = [r for r in rows if r["dataset"]==ds and r["config"]==tag and r["status"]=="ok"]
        if not sub: continue
        a = dict(dataset=ds, config=tag, n=len(sub))
        for k in ("mean_auc","mean_auprc","mean_acc","mean_f1","top1","H_s","ratio"):
            v=[r[k] for r in sub if k in r]
            if v: a[f"{k}_mean"]=np.mean(v); a[f"{k}_std"]=np.std(v, ddof=1) if len(v)>1 else 0.0
        agg.append(a)
acols=["dataset","config","n"]+[f"{k}_{s}" for k in ("mean_auc","mean_auprc","mean_acc","mean_f1","top1","H_s","ratio") for s in ("mean","std")]
with open(OUT/"final_grid_mean_std.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=acols,extrasaction="ignore"); w.writeheader()
    for a in agg: w.writerow(a)

print(f"per-seed: {len(rows)} rows ({sum(1 for r in rows if r['status']=='ok')} ok)")
for a in agg:
    print(f"{a['dataset']:7s} {a['config']:9s} n={a['n']}  "
          f"AUC {a.get('mean_auc_mean',0):.4f}±{a.get('mean_auc_std',0):.4f}  "
          f"AUPRC {a.get('mean_auprc_mean',0):.4f}±{a.get('mean_auprc_std',0):.4f}  "
          f"top1 {a.get('top1_mean',0):.3f}  ratio {a.get('ratio_mean',0):.2f}")

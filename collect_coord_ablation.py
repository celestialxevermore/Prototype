#!/usr/bin/env python
"""
collect_coord_ablation.py
=======================================================================
coord_ablation_entry.py 가 남긴 profile.json 들을 모아서 표로 정리.

    python collect_coord_ablation.py [--root <PROF_ROOT>] [--csv]

출력
  [T1] 방식별 요약   : 파라미터 수 / pretrain 시간 / peak mem / alignment ms/batch
  [T2] alignment 비중: phase 별 align median, step median, share(%)
  [T3] 성능          : multi-source AUROC(6 sources) / zero-shot / few-shot(4~64)
  [T4] 총 wall-clock : few_shot 별 end-to-end 시간
"""
import argparse
import json
import os
import re
import statistics as st

MODES = ["cos", "xattn", "fgw"]
MODE_LABEL = {
    "cos":   "1. Pooled CLS cosine",
    "xattn": "2. Cross-attention",
    "fgw":   "3. FGW (Ours)",
}
SHOTS = [0, 4, 8, 16, 32, 64]
PRETRAIN_PHASES = ("phase1_vanilla_gat", "bridge_lcg_init", "phase2_joint")


def load(root):
    out = {}
    for m in MODES:
        for s in SHOTS:
            p = os.path.join(root, m, f"shot{s}", "profile.json")
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        out[(m, s)] = json.load(f)
                except Exception as e:
                    print(f"  ! parse fail {p}: {e}")
    return out


def phase_sec(prof, names):
    tot, seen = 0.0, False
    for ph in prof.get("phases", []):
        if ph.get("phase") in names:
            tot += float(ph.get("sec", 0.0)); seen = True
    return tot if seen else None


def peak_mem(prof):
    vals = [float(ph.get("peak_alloc_GB", 0.0)) for ph in prof.get("phases", [])]
    vals.append(float((prof.get("gpu_peak_overall") or {}).get("peak_alloc_GB", 0.0)))
    return max(vals) if vals else None


def align_ms(prof, prefer=("phase2_joint", "fewshot_adapt")):
    w = (prof.get("align_timer") or {}).get("windows") or {}
    for k in prefer:
        if k in w and w[k].get("median_ms"):
            return w[k]["median_ms"], k
    for k, v in w.items():
        if v.get("median_ms"):
            return v["median_ms"], k
    return None, None


def step_ms(prof, phase):
    w = (prof.get("step_timer") or {}).get("windows") or {}
    v = w.get(phase) or {}
    return v.get("median_ms")


def params(prof):
    p = prof.get("params") or {}
    rec = p.get("Full") or p.get("Few") or (p if "align_params_total" in p else {})
    return rec


def fmt(v, n=4, dash="-"):
    return dash if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def _rows(prof, key):
    """metrics[key] -> [(phase, [groups...])]  (구/신 포맷 모두 허용)"""
    out = []
    for r in ((prof.get("metrics") or {}).get(key) or []):
        if isinstance(r, dict):
            out.append((r.get("phase", "?"), r.get("g", [])))
        else:
            out.append(("?", list(r)))
    return out


def best_source_report(prof):
    """Phase 2 epoch 중 Global mean AUC 최고 지점 (= best ckpt 선택 기준)."""
    rows = [r for r in _rows(prof, "pre_auc") if r[0] in ("phase2_joint", "?")]
    if not rows:
        return None
    ph, g = max(rows, key=lambda r: float(r[1][4]))
    epoch, total, l_mean, l_per, g_mean, g_per = g
    parse = lambda s: [float(x.strip().strip("'\"")) for x in s.split(",") if x.strip()]
    return {"epoch": int(epoch), "local_mean": float(l_mean), "global_mean": float(g_mean),
            "local_per": parse(l_per), "global_per": parse(g_per)}


# fgw(원본 LCG.py) 는 "[ROUTING]", ablation quantizer 는 "[ROUTING/cos]" 처럼 찍는다.
_ROUTING = re.compile(
    r"\[ROUTING(?:/\w+)?\].*?H_s=([\d.]+) H_p=([\d.]+) \| top1=([\d.]+)")


def routing_entropy(root):
    """shot0 로그의 ROUTING 라인에서 H_p / H_s / top1 median 추출."""
    out = {}
    for m in MODES:
        f = os.path.join(root, "logs", f"{m}_shot0.log")
        if not os.path.exists(f):
            continue
        hs, hp, t1 = [], [], []
        try:
            with open(f, errors="replace") as fh:
                for line in fh:
                    mt = _ROUTING.search(line)
                    if mt:
                        hs.append(float(mt.group(1)))
                        hp.append(float(mt.group(2)))
                        t1.append(float(mt.group(3)))
        except Exception:
            continue
        if hs:
            out[m] = {"H_s": st.median(hs), "H_p": st.median(hp),
                      "top1": st.median(t1), "n": len(hs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/storage/personal/eungyeop/experiments/"
                                      "experiments/coord_ablation_20260727")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    P = load(a.root)
    if not P:
        print(f"no profile.json under {a.root}")
        return
    print(f"# coordinate ablation summary   (root = {a.root})")
    print(f"# collected runs: {len(P)}\n")

    # ---------------- T1: 방식별 요약 -----------------------------------
    print("## [T1] 방식별 요약")
    hdr = ("| coordinate | align params | model params | pretrain (s) | "
           "peak GPU mem (GB) | align median (ms/batch) |")
    print(hdr); print("|" + "---|" * 6)
    for m in MODES:
        base = P.get((m, 0)) or next((P[(m, s)] for s in SHOTS if (m, s) in P), None)
        if base is None:
            print(f"| {MODE_LABEL[m]} | - | - | - | - | - |"); continue
        pr = params(base)
        pre = phase_sec(base, PRETRAIN_PHASES)
        mem = max([peak_mem(P[(m, s)]) or 0 for s in SHOTS if (m, s) in P] or [0])
        ams, _ = align_ms(base)
        print(f"| {MODE_LABEL[m]} | {pr.get('align_params_total', '-'):,} | "
              f"{pr.get('model_params_total', 0):,} | {fmt(pre, 1)} | "
              f"{fmt(mem, 3)} | {fmt(ams, 3)} |")

    # ---------------- T2: alignment 비중 --------------------------------
    print("\n## [T2] alignment 연산 비중 (forward only, warm-up 10 / median of 50)")
    print("| coordinate | phase | align median (ms) | full step median (ms) | share (%) |")
    print("|" + "---|" * 5)
    for m in MODES:
        for s in SHOTS:
            prof = P.get((m, s))
            if not prof:
                continue
            for ph, v in sorted((prof.get("align_share") or {}).items()):
                print(f"| {MODE_LABEL[m]} | {ph} (shot={s}) | "
                      f"{v['align_median_ms']:.3f} | {v['step_median_ms']:.3f} | "
                      f"{v['align_share_pct']:.2f} |")

    # ---------------- T3: 성능 -------------------------------------------
    print("\n## [T3] 성능  (target = heart)")
    print("| coordinate | multi-source AUROC (LCG / GAT) | zero-shot AUC | "
          + " | ".join(f"{s}-shot AUC" for s in SHOTS if s > 0) + " |")
    print("|" + "---|" * (3 + len([s for s in SHOTS if s > 0])))
    src_detail = {}
    for m in MODES:
        msrc, rep = "-", None
        for s in SHOTS:
            if (m, s) in P:
                rep = best_source_report(P[(m, s)])
                if rep:
                    break
        if rep:
            src_detail[m] = rep
            msrc = f"{rep['global_mean']:.4f} / {rep['local_mean']:.4f}"
        zs, z = "-", P.get((m, 0))
        if z:
            zz = _rows(z, "zero_shot")
            if zz:
                zs = f"{float(zz[-1][1][0]):.4f}"
        cells = []
        for s in SHOTS:
            if s == 0:
                continue
            prof, c = P.get((m, s)), "-"
            if prof:
                fs = _rows(prof, "few_shot_summary")
                if fs:
                    g = fs[-1][1]
                    c = f"{float(g[1]):.4f}±{float(g[2]):.4f}"
                else:
                    per = _rows(prof, "few_shot")
                    if per:
                        aucs = [float(x[1][2]) for x in per]
                        c = (f"{st.mean(aucs):.4f}±{st.pstdev(aucs):.4f}"
                             if len(aucs) > 1 else f"{aucs[0]:.4f}")
            cells.append(c)
        print(f"| {MODE_LABEL[m]} | {msrc} | {zs} | " + " | ".join(cells) + " |")

    # ---------------- T3b: 6개 소스별 AUROC -------------------------------
    if src_detail:
        print("\n## [T3b] multi-source per-source AUROC (best Phase-2 epoch, Global/LCG)")
        srcs = ["Medical", "CardioDis", "Statlog", "Erbil", "SAheart", "HF-clinical"]
        print("| coordinate | best ep | " + " | ".join(srcs) + " | mean |")
        print("|" + "---|" * (len(srcs) + 3))
        for m in MODES:
            r = src_detail.get(m)
            if not r:
                print(f"| {MODE_LABEL[m]} |" + " - |" * (len(srcs) + 2)); continue
            per = r["global_per"] + [None] * (len(srcs) - len(r["global_per"]))
            print(f"| {MODE_LABEL[m]} | {r['epoch']} | "
                  + " | ".join(fmt(v) for v in per[:len(srcs)])
                  + f" | {r['global_mean']:.4f} |")

    # ---------------- T4: wall-clock -------------------------------------
    print("\n## [T4] end-to-end wall-clock (초)")
    print("| coordinate | " + " | ".join(f"shot={s}" for s in SHOTS) + " | 합계 |")
    print("|" + "---|" * (len(SHOTS) + 2))
    for m in MODES:
        cells, tot = [], 0.0
        for s in SHOTS:
            prof = P.get((m, s))
            v = prof.get("total_sec") if prof else None
            if v:
                tot += float(v)
            cells.append(fmt(float(v), 1) if v else "-")
        print(f"| {MODE_LABEL[m]} | " + " | ".join(cells) + f" | {tot:.1f} |")

    # ---------------- T5: 상세 phase 시간 ---------------------------------
    print("\n## [T5] phase 별 시간 / peak mem")
    print("| coordinate | shot | phase | sec | peak alloc (GB) |")
    print("|" + "---|" * 5)
    for m in MODES:
        for s in SHOTS:
            prof = P.get((m, s))
            if not prof:
                continue
            for ph in prof.get("phases", []):
                print(f"| {MODE_LABEL[m]} | {s} | {ph.get('phase')} | "
                      f"{float(ph.get('sec', 0)):.1f} | "
                      f"{float(ph.get('peak_alloc_GB', 0)):.3f} |")

    # ---------------- T6: routing entropy ---------------------------------
    #   τ 를 세 방식에 동일하게 두면 score scale 차이 때문에 routing sharpness 가
    #   달라진다. baseline 이 불리한 게 정렬 방식 때문인지 온도 때문인지 구분용.
    ent = routing_entropy(a.root)
    if ent:
        print("\n## [T6] routing entropy (Phase 2, ROUTING 로그 median)")
        print("| coordinate | H_p (prototype usage) | H_s (per-sample) | top1 | n |")
        print("|" + "---|" * 5)
        for m in MODES:
            e = ent.get(m)
            if not e:
                print(f"| {MODE_LABEL[m]} | - | - | - | - |"); continue
            print(f"| {MODE_LABEL[m]} | {e['H_p']:.3f} | {e['H_s']:.3f} | "
                  f"{e['top1']:.3f} | {e['n']} |")

    # ---------------- freeze check ----------------------------------------
    print("\n## [체크] target adaptation 시 alignment param frozen")
    for m in MODES:
        vals = set()
        for s in SHOTS:
            prof = P.get((m, s))
            if prof:
                for v in prof.get("freeze_check", []):
                    vals.add(str(v))
        print(f"  - {MODE_LABEL[m]}: {sorted(vals) or '-'}")

    if a.csv:
        out = os.path.join(a.root, "coord_ablation_summary.csv")
        with open(out, "w") as f:
            f.write("mode,shot,total_sec,pretrain_sec,peak_mem_GB,align_median_ms,"
                    "step_median_ms,align_share_pct,align_params,model_params\n")
            for (m, s), prof in sorted(P.items()):
                ams, ph = align_ms(prof)
                sms = step_ms(prof, ph) if ph else None
                sh = (prof.get("align_share") or {}).get(ph, {}).get("align_share_pct")
                pr = params(prof)
                f.write(f"{m},{s},{fmt(prof.get('total_sec'),1)},"
                        f"{fmt(phase_sec(prof, PRETRAIN_PHASES),1)},"
                        f"{fmt(peak_mem(prof),3)},{fmt(ams,3)},{fmt(sms,3)},"
                        f"{fmt(sh,2)},{pr.get('align_params_total','')},"
                        f"{pr.get('model_params_total','')}\n")
        print(f"\ncsv -> {out}")


if __name__ == "__main__":
    main()

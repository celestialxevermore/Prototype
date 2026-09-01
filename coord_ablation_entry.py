#!/usr/bin/env python
"""
coord_ablation_entry.py
=======================================================================
main_SS.py 를 **한 줄도 고치지 않고** coordinate ablation 으로 돌리는 진입점.

    python coord_ablation_entry.py --coord_mode {fgw,cos,xattn} \
        --profile_out <dir> -- <main_SS.py 인자 그대로...>

하는 일 (전부 runtime monkey-patch, 소스 파일 변경 없음):
  1) models.TabularFLM_S_.GraphQuantizer  -> LCG_ablation.build_quantizer
  2) Model.__init__ 래핑                  -> args 에 coord_* 주입 + quantizer 에
                                             parent 참조 붙여 CLS 접근 가능하게
  3) utils.train_test.binary_train/multi_train 래핑
                                          -> 전체 학습 step 시간 측정
  4) "my_experiment_logger" 에 핸들러 부착 -> phase 경계 감지
                                             (시간 / peak GPU mem / 타이머 rearm)
  5) atexit -> profile.json 덤프

측정 프로토콜
  - alignment / step 둘 다 구간 앞뒤 torch.cuda.synchronize() + time.perf_counter()
  - warm-up 10 batch 폐기 후 50 batch 의 median
  - alignment 는 forward 만 (backward 제외), step 은 forward+backward+optimizer
  - 창이 차면 자동 비활성 → 이후 batch 에는 sync 오버헤드 없음 (총 시간 왜곡 방지)
"""

import argparse
import atexit
import json
import logging
import os
import re
import runpy
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from models.LCG_ablation import (  # noqa: E402
    WindowTimer, assert_alignment_frozen, build_quantizer, coord_param_report,
    normalize_coord_mode,
)

# =====================================================================
# 0) 우리 전용 인자 분리
# =====================================================================
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--coord_mode', required=True, choices=['fgw', 'cos', 'xattn'])
_ap.add_argument('--profile_out', required=True, help='profile.json 저장 디렉토리')
_ap.add_argument('--coord_align_tau', type=float, default=None,
                 help='coordinate softmax 온도. 미지정 시 --soft_tau 와 동일 (통제 조건)')
_ap.add_argument('--coord_attn_dim', type=int, default=None)
_ap.add_argument('--warmup_batches', type=int, default=10)
_ap.add_argument('--measure_batches', type=int, default=50)
KNOWN, PASSTHROUGH = _ap.parse_known_args()
PASSTHROUGH = [a for a in PASSTHROUGH if a != '--']   # 구분자 제거
COORD_MODE = normalize_coord_mode(KNOWN.coord_mode)
os.makedirs(KNOWN.profile_out, exist_ok=True)

PROFILE = {
    "coord_mode": COORD_MODE,
    "argv": PASSTHROUGH,
    "warmup_batches": KNOWN.warmup_batches,
    "measure_batches": KNOWN.measure_batches,
    "phases": [],          # [{phase, start, end, sec, peak_alloc_GB, peak_reserved_GB}]
    "params": {},
    "align_timer": {},
    "step_timer": {},
    "metrics": {},
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# 전체 학습 step 타이머 (forward+backward+optimizer)
STEP_TIMER = WindowTimer("train_step", warmup=KNOWN.warmup_batches,
                         n_measure=KNOWN.measure_batches, enabled=True)

_MODELS = {}          # {'Full': Model, 'Few': Model}
_FREEZE_CHECK = []    # xattn: 타겟에서 W_Q/W_K/W_V 고정 여부 검증 결과


# =====================================================================
# 1) quantizer 교체 + Model.__init__ 래핑
# =====================================================================
import models.TabularFLM_S_ as _TF  # noqa: E402


def _quantizer_factory(args, *a, **kw):
    args.coord_mode = COORD_MODE
    args.coord_warmup_batches = KNOWN.warmup_batches
    args.coord_measure_batches = KNOWN.measure_batches
    args.coord_time_profile = True
    if KNOWN.coord_align_tau is not None:
        args.coord_align_tau = KNOWN.coord_align_tau
    if KNOWN.coord_attn_dim is not None:
        args.coord_attn_dim = KNOWN.coord_attn_dim
    return build_quantizer(args, *a, **kw)


_TF.GraphQuantizer = _quantizer_factory

_orig_model_init = _TF.Model.__init__


def _patched_model_init(self, args, *a, **kw):
    args.coord_mode = COORD_MODE
    args.coord_warmup_batches = KNOWN.warmup_batches
    args.coord_measure_batches = KNOWN.measure_batches
    args.coord_time_profile = True
    if KNOWN.coord_align_tau is not None:
        args.coord_align_tau = KNOWN.coord_align_tau
    if KNOWN.coord_attn_dim is not None:
        args.coord_attn_dim = KNOWN.coord_attn_dim
    _orig_model_init(self, args, *a, **kw)
    q = getattr(self, "graph_quantizer", None)
    if q is not None and hasattr(q, "attach_parent"):
        q.attach_parent(self)          # CLS(x_basis[:,0,:]) 접근용
    _MODELS[str(getattr(self, "mode", "?"))] = self


_TF.Model.__init__ = _patched_model_init

# --- 타겟 adapt 시 alignment 파라미터가 진짜 얼어 있는지 검증 -------------
_orig_freeze = _TF.Model.set_freeze_target


def _patched_freeze(self, *a, **kw):
    out = _orig_freeze(self, *a, **kw)
    try:
        _FREEZE_CHECK.append(assert_alignment_frozen(self, strict=False))
    except Exception as e:
        _FREEZE_CHECK.append({"error": str(e)})
    return out


_TF.Model.set_freeze_target = _patched_freeze


# =====================================================================
# 2) train step 시간 측정 (utils.train_test 래핑)
# =====================================================================
import utils.train_test as _TT  # noqa: E402


def _make_timed_train(orig_fn, is_multi=False):
    def _timed_train(model, train_loader, criterion, optimizer, device):
        if not STEP_TIMER.active:
            return orig_fn(model, train_loader, criterion, optimizer, device)
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            with STEP_TIMER.measure():
                optimizer.zero_grad()
                loss = model(batch, batch['y'])
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(batch['y'])
        return total_loss / len(train_loader.dataset)
    return _timed_train


_TT.binary_train = _make_timed_train(_TT.binary_train)
_TT.multi_train = _make_timed_train(_TT.multi_train, is_multi=True)


# =====================================================================
# 3) phase 경계 감지 (로그 핸들러) + peak mem
# =====================================================================
_PHASE_MARKERS = [
    ("phase1_vanilla_gat", re.compile(r"\[Phase 1\] Start Vanilla GAT")),
    ("bridge_lcg_init",    re.compile(r"\[Bridge\] Initializing LCG")),
    ("phase2_joint",       re.compile(r"\[Phase 2\] Start Joint Training")),
    ("source_report",      re.compile(r"\[Full\] Using loaded pretrain")),
    ("target_load",        re.compile(r"^\[Target\] target = ")),
    ("zeroshot_eval",      re.compile(r"\[Zero-shot\] Evaluating")),
    ("fewshot_adapt",      re.compile(r"\[Few-shot\] support resamples R")),
]

_METRIC_PATTERNS = {
    "zero_shot": re.compile(
        r"\[Zero-shot\] Test Results: AUC=([\d.]+) AUPRC=([\d.]+) ACC=([\d.]+) "
        r"Prec=([\d.]+) Rec=([\d.]+) F1=([\d.]+)"),
    "few_shot": re.compile(
        r"\[Few-shot\]\[Ep (\d+)/(\d+)\] AUC=([\d.]+) AUPRC=([\d.]+) ACC=([\d.]+) "
        r"Prec=([\d.]+) Rec=([\d.]+) F1=([\d.]+)"),
    # 소스 pretrain epoch 리포트 (multi-line 한 레코드) — epoch / local / global 동시 캡처
    "pre_auc": re.compile(
        r"\[Pre\]\[Epoch (\d+)/(\d+)\]\s*"
        r">>> Local \(GAT\): Mean AUC ([\d.]+) \| Per-Source: \[([^\]]*)\]\s*"
        r">>> Global\(LCG\): Mean AUC ([\d.]+) \| Per-Source: \[([^\]]*)\]", re.S),
    "pre_auprc": re.compile(
        r"\[Pre\]\[Epoch (\d+)/(\d+)\]\s*"
        r">>> Local \(GAT\): Mean AUPRC ([\d.]+) \| Per-Source: \[([^\]]*)\]\s*"
        r">>> Global\(LCG\): Mean AUPRC ([\d.]+) \| Per-Source: \[([^\]]*)\]", re.S),
    "few_shot_summary": re.compile(
        r"\[Few-shot\]\[Summary\] R=(\d+) resamples \| "
        r"AUC=([\d.]+)±([\d.]+) AUPRC=([\d.]+)±([\d.]+) ACC=([\d.]+)±([\d.]+) "
        r"Prec=([\d.]+)±([\d.]+) Rec=([\d.]+)±([\d.]+) F1=([\d.]+)±([\d.]+)"),
}


def _gpu_peak():
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_GB": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_GB": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }


class _PhaseTracker(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.cur = None
        self.t0 = time.time()

    def _close_phase(self):
        if self.cur is None:
            return
        rec = self.cur
        rec["sec"] = time.time() - rec["_t0"]
        rec.update(_gpu_peak())
        rec.pop("_t0", None)
        PROFILE["phases"].append(rec)
        self.cur = None

    def open_phase(self, name):
        self._close_phase()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.cur = {"phase": name, "_t0": time.time()}
        # phase 마다 새 측정 창
        STEP_TIMER.rearm(name)
        for mode, mm in _MODELS.items():
            q = getattr(mm, "graph_quantizer", None)
            if q is not None and hasattr(q, "rearm_timer"):
                q.rearm_timer(name)

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        for name, pat in _PHASE_MARKERS:
            if pat.search(msg):
                self.open_phase(name)
                break
        for key, pat in _METRIC_PATTERNS.items():
            mt = pat.search(msg)
            if mt:
                cur = self.cur["phase"] if self.cur else "?"
                PROFILE["metrics"].setdefault(key, []).append(
                    {"phase": cur, "g": list(mt.groups())})


_TRACKER = _PhaseTracker()
for _name in ("my_experiment_logger", "utils.util", ""):
    logging.getLogger(_name).addHandler(_TRACKER)


# =====================================================================
# 4) 종료 시 profile.json 덤프
# =====================================================================
def _finalize():
    _TRACKER._close_phase()
    PROFILE["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    PROFILE["total_sec"] = time.time() - _TRACKER.t0
    PROFILE["step_timer"] = STEP_TIMER.report()

    # 파라미터 수 / alignment 타이머 (Full=source pretrain, Few=target adapt)
    merged_windows, params = {}, {}
    for mode, model in _MODELS.items():
        try:
            params[mode] = coord_param_report(model)
        except Exception as e:
            params[mode] = {"error": str(e)}
        q = getattr(model, "graph_quantizer", None)
        if q is not None and hasattr(q, "timing_report"):
            try:
                rep = q.timing_report()
                PROFILE.setdefault("align_timer_by_model", {})[mode] = rep
                for ph, w in (rep.get("windows") or {}).items():
                    if w.get("n_measured"):
                        merged_windows.setdefault(ph, w)
            except Exception as e:
                PROFILE.setdefault("align_timer_by_model", {})[mode] = {"error": str(e)}
    PROFILE["params"] = params
    PROFILE["align_timer"] = {"coord_mode": COORD_MODE, "windows": merged_windows}
    PROFILE["freeze_check"] = _FREEZE_CHECK

    # alignment 비중(%) = align median / step median  (같은 phase 창끼리)
    share = {}
    aw = merged_windows
    sw = (PROFILE.get("step_timer") or {}).get("windows") or {}
    for ph in set(aw) & set(sw):
        a_med, s_med = aw[ph].get("median_ms"), sw[ph].get("median_ms")
        if a_med and s_med:
            share[ph] = {
                "align_median_ms": a_med,
                "step_median_ms": s_med,
                "align_share_pct": 100.0 * a_med / s_med,
            }
    PROFILE["align_share"] = share
    PROFILE["gpu_peak_overall"] = _gpu_peak()

    out = os.path.join(KNOWN.profile_out, "profile.json")
    try:
        with open(out, "w") as f:
            json.dump(PROFILE, f, indent=2, default=str)
        print(f"\n[coord_ablation] profile saved -> {out}")
        for ph, v in share.items():
            print(f"  [{ph}] align={v['align_median_ms']:.3f} ms | "
                  f"step={v['step_median_ms']:.3f} ms | share={v['align_share_pct']:.2f}%")
    except Exception as e:
        print(f"[coord_ablation] profile save FAILED: {e}")


atexit.register(_finalize)


# =====================================================================
# 5) main_SS.py 실행
# =====================================================================
if __name__ == "__main__":
    sys.argv = ["main_SS.py"] + PASSTHROUGH
    print(f"[coord_ablation] coord_mode={COORD_MODE}")
    print(f"[coord_ablation] argv={' '.join(sys.argv)}")
    runpy.run_path(os.path.join(REPO, "main_SS.py"), run_name="__main__")

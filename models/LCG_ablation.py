"""
models/LCG_ablation.py
=======================================================================
Coordinate(=routing) ablation modules for ProtoLLM.

목적: instance graph G^(m) 와 prototype graph Ḡ_k 를 "어떻게 정렬해서
coordinate c 를 뽑는가" 만 바꾼다. 나머지는 전부 동일하게 둔다.

  - GAT encoder (BasisGATLayer)            : 그대로
  - prototype graph (LatentCompositeGraph)  : 그대로 (M, K, D 동일)
  - basis GNN (LatentCompositeGNN)          : 그대로
  - 예측 경로 (ghead / thead / sheads)      : 그대로
  - τ (soft_tau), K (n_nodes), T(=K)        : 그대로
  - reg_loss = (1-H_p)^2 + hs_reg * H_s     : 그대로 (pi 에만 의존)
  - 반환 시그니처                            : 그대로 (drop-in)

바뀌는 것은 오직 pi(=coordinate) 를 만드는 scoring 함수 하나뿐이다.

    baseline (fgw)  : c = softmax(-FGW(G^(m), Ḡ_k) / τ)
    ablation (cos)  : c = softmax( cos(z^(m), ḡ_k) / τ )      # ḡ_k = mean_T(F̄_k)
    ablation (xattn): c = softmax( s_k / τ )                   # cross-attention scorer

세 모드 모두 score 는 "클수록 가깝다" 규약으로 통일되어 있고, 동일하게
`/ τ` 후 softmax 를 태운다. (fgw 는 -distance 이므로 규약 일치)

---------------------------------------------------------------------
공정성 관련 주의 (논문에 같이 적어야 하는 것)
---------------------------------------------------------------------
* FGW 는 학습 파라미터가 0개다. cross-attention 은 W_Q/W_K/W_V + scorer 가
  추가되므로 엄밀한 통제가 아니다. → `param_report()` 로 파라미터 수를
  반드시 같이 리포트할 것.
* cross-attention 의 W_Q/W_K/W_V 는 source 에서 학습하고 target 에서는 고정.
  `Model.set_freeze_target()` 이 이미 basis/thead/ghead 외 전부 freeze 하므로
  자동으로 고정되지만, `assert_alignment_frozen(model)` 로 검증할 것.
* cos ablation 은 A^(m), Ā_k (쌍별 구조) 를 아예 쓰지 않는다. xattn 은 노드
  속성만 쓰고 쌍별 구조는 쓰지 않는다. 그게 이 비교가 보이려는 지점이다.

---------------------------------------------------------------------
Wall-clock
---------------------------------------------------------------------
`AlignTimer` 가 alignment 연산 구간만 잰다 (ms/batch).
CUDA 에서는 cuda.Event 를 쓰되, elapsed_time() 을 몇 스텝 늦게 resolve 해서
매 배치 synchronize 로 총 학습시간이 왜곡되는 것을 피한다.
총 학습시간은 main_SS.py 가 이미 `start_time` / `elapsed_sec` 로 기록 중.

    q = model.graph_quantizer
    q.timing_report()   # {'n_batches', 'align_ms_mean', 'align_ms_median', ...}
    q.param_report()    # {'coord_mode', 'align_params_total', ...}
    q.reset_timing()    # phase 경계에서 리셋 (Phase1/Phase2/few-shot 분리 측정)

---------------------------------------------------------------------
측정 프로토콜 (리포트해야 할 셀)
---------------------------------------------------------------------
    for mode in [fgw, cos, xattn]:
        (a) multi-source AUROC  : 6개 source val AUROC (pretrain 리포트)
        (b) zero-shot           : --few_shot 0
        (c) 4-shot / 8-shot     : --few_shot 4 / --few_shot 8
        (d) align ms/batch      : q.timing_report()
        (e) 총 학습 시간         : results['elapsed_sec']
        (f) 파라미터 수          : coord_param_report(model)
"""

import contextlib
import logging
import math
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # POT batch API 가 없는 환경에서도 cos/xattn 은 쓸 수 있게
    from models.LCG import GraphQuantizer
    _HAS_FGW = True
except Exception as _e:  # pragma: no cover
    GraphQuantizer = object
    _HAS_FGW = False
    _FGW_IMPORT_ERROR = _e

logger = logging.getLogger("my_experiment_logger")

COORD_MODES = ("fgw", "cos", "xattn")


# =====================================================================
# Wall-clock timer
# =====================================================================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WindowTimer:
    """구간 측정 타이머 (측정 프로토콜 고정).

      - 구간 앞뒤에 torch.cuda.synchronize() + time.perf_counter()
      - warm-up `warmup` batch 는 버리고, 이후 `n_measure` batch 를 수집
      - 창이 차면 자동으로 비활성 → 이후 batch 는 sync 비용 0 (총 학습시간 왜곡 방지)
      - `rearm(phase)` 로 phase 마다 새 창을 열 수 있음 (phase 별 stats 보관)

    보고값은 median (mean/p90/min/max 도 같이 남김).
    """

    def __init__(self, name: str, warmup: int = 10, n_measure: int = 50,
                 enabled: bool = True):
        self.name = str(name)
        self.warmup = int(warmup)
        self.n_measure = int(n_measure)
        self.enabled = bool(enabled)
        self.windows = {}          # phase -> stats dict
        self._phase = "init"
        self._seen = 0
        self._samples = []
        self.last_ms = float("nan")

    # 창이 아직 안 찼을 때만 True → 이때만 synchronize 가 걸린다
    @property
    def active(self) -> bool:
        return self.enabled and len(self._samples) < self.n_measure

    def rearm(self, phase: str):
        """phase 전환. 이전 창은 확정 저장하고 새 창을 연다."""
        self._flush()
        self._phase = str(phase)
        self._seen = 0
        self._samples = []

    def _flush(self):
        if not self._samples:
            return
        xs = sorted(self._samples)
        n = len(xs)
        self.windows[self._phase] = {
            "n_warmup_discarded": self.warmup,
            "n_measured": n,
            "median_ms": statistics.median(xs),
            "mean_ms": sum(xs) / n,
            "p90_ms": xs[min(n - 1, int(0.9 * (n - 1)))],
            "min_ms": xs[0],
            "max_ms": xs[-1],
        }

    @contextlib.contextmanager
    def measure(self):
        if not self.active:
            yield
            return
        cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            dt = (time.perf_counter() - t0) * 1e3
            self.last_ms = dt
            self._seen += 1
            if self._seen > self.warmup:      # warm-up 10 batch 폐기
                self._samples.append(dt)

    def report(self):
        self._flush()
        return {"timer": self.name, "warmup": self.warmup,
                "n_measure": self.n_measure, "windows": dict(self.windows)}

    def reset(self):
        self.windows = {}
        self._phase = "init"
        self._seen = 0
        self._samples = []
        self.last_ms = float("nan")


# 하위호환 별칭
AlignTimer = WindowTimer


# =====================================================================
# Base: GraphQuantizer 와 동일한 인터페이스 / 동일한 reg_loss
# =====================================================================
class _BaseCoordQuantizer(nn.Module):
    """GraphQuantizer drop-in base.

    하위 클래스는 `compute_scores(...) -> [B, M]` 하나만 구현하면 된다.
    (클수록 가깝다 규약)
    """

    COORD_MODE = "base"
    USES_STRUCTURE = False  # A^(m) / Ā_k 를 쓰는가

    def __init__(self, args, alpha=0.5, tau=0.1, eps=0.05,
                 outer_iters=20, sinkhorn_iters=200):
        super().__init__()
        self.args = args

        # --- GraphQuantizer 와 이름/의미 동일하게 유지 ---
        self.alpha = alpha
        self.reg = getattr(args, "reg", 0.01)
        self.vq_beta = getattr(args, "vq_beta", 0.3)
        self.tau = getattr(args, "tau", tau)
        self.soft_tau = float(args.soft_tau)
        self.orth_reg = getattr(args, "orth_reg", 0.1)
        self.div_reg = getattr(args, "div_reg", 0.0)
        self.ent_reg = getattr(args, "entropy_reg", 0.01)

        # coordinate softmax temperature.
        # 기본은 baseline(FGW) 과 **동일한 soft_tau** — "τ 그대로" 통제 조건.
        # NOTE: main_SS.py 에 이미 --coord_tau (CoordinatorMLP 용, default 0.3) 가
        #       있으므로 이름 충돌을 피해 `coord_align_tau` 로 override 한다.
        _ct = getattr(args, "coord_align_tau", None)
        self.coord_tau = float(self.soft_tau) if _ct in (None, 0) else float(_ct)

        self.logger = logging.getLogger("my_experiment_logger")
        self.logger_name = "my_experiment_logger"
        self.log_interval = 50
        self.current_epoch = 0
        self.phase2_start_epoch = None

        # main_SS.py / TabularFLM_S_.py 가 참조하는 상태들
        self.last_pi = None
        self.last_plan = None  # alignment plan 없음 → 항상 None
        self._last_H_p = float("nan")
        self._last_H_s = float("nan")

        self.register_buffer("has_printed_initial_weights",
                             torch.tensor(False), persistent=False)
        self.register_buffer("log_step", torch.tensor(0), persistent=False)

        M = int(args.n_graphs)
        self.register_buffer("usage_count", torch.zeros(M))
        self.register_buffer("reset_counter", torch.tensor(0))
        self.dead_threshold = 0.05
        self.reset_noise = 0.01

        self.timer = WindowTimer(
            name=f"align/{self.COORD_MODE}",
            warmup=int(getattr(args, "coord_warmup_batches", 10)),
            n_measure=int(getattr(args, "coord_measure_batches", 50)),
            enabled=bool(getattr(args, "coord_time_profile", True)),
        )
        self._parent_ref = None  # CLS 접근용 (weakref, submodule 등록 안 됨)

    # ---------------- parent Model 참조 (cls_feat 자동 획득) ----------------
    def attach_parent(self, model):
        """Model.predict 가 self.x_basis 를 세팅한 뒤 quantizer 를 호출하므로,
        parent 참조만 있으면 CLS(x_basis[:,0,:]) 를 별도 인자 없이 읽을 수 있다.
        nn.Module 속성으로 넣으면 submodule 로 등록되어 재귀하므로 weakref 사용."""
        import weakref
        object.__setattr__(self, "_parent_ref", weakref.ref(model))

    def _resolve_cls(self, cls_feat, source_feat):
        if cls_feat is not None:
            return cls_feat
        ref = getattr(self, "_parent_ref", None)
        parent = ref() if ref is not None else None
        xb = getattr(parent, "x_basis", None) if parent is not None else None
        if torch.is_tensor(xb) and xb.shape[0] == source_feat.shape[0]:
            return xb[:, 0, :]
        return None

    # ---------------- 호환용 no-op (baseline 과 동일 동작) ----------------
    def reset_dead(self, latent_graph):
        """baseline GraphQuantizer.reset_dead 와 동일한 동작.

        usage_count 가 forward 에서 갱신되지 않으므로 baseline 에서도 항상
        'No dead codes' 로 빠진다. 동작을 일부러 그대로 맞춰둔다.
        """
        M = self.usage_count.shape[0]
        avg_usage = self.usage_count.sum() / M
        dead_mask = self.usage_count < (avg_usage * self.dead_threshold)
        if not dead_mask.any():
            self.logger.info(
                f"   [Dead Reset] No dead codes. Usage: {self.usage_count.cpu().numpy().round(1)}"
            )
            self.usage_count.zero_()
            return 0
        dead_indices = dead_mask.nonzero(as_tuple=True)[0].tolist()
        self.logger.info(f"   [Dead Reset] Dead: {dead_indices}")
        with torch.no_grad():
            for d_idx in dead_indices:
                if hasattr(latent_graph, "init_node_embeddings"):
                    latent_graph.node_embeddings.data[d_idx] = \
                        latent_graph.init_node_embeddings[d_idx].clone()
        self.usage_count.zero_()
        return len(dead_indices)

    def get_diversifying_loss(self, coordinates, y):
        distance = (coordinates.unsqueeze(1) - coordinates.unsqueeze(0)).abs().sum(dim=2)
        label_similarity = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        return (distance * label_similarity).sum() / distance.sum().clamp_min(1e-8)

    # ---------------- routing regularizer (baseline 과 동일 수식) ----------------
    def _routing_reg(self, pi):
        M = pi.shape[1]
        H_max = math.log(M)
        p = pi.mean(dim=0)
        H_p = -(p * (p + 1e-9).log()).sum() / H_max
        pi_safe = pi.clamp_min(1e-12)
        H_s = -(pi_safe * pi_safe.log()).sum(dim=1).mean() / H_max
        hs_coeff = getattr(self.args, "hs_reg", 0.1)
        reg_loss = 1.0 * (1 - H_p) ** 2 + hs_coeff * H_s
        return reg_loss, H_p, H_s

    # ---------------- 하위 클래스 구현부 ----------------
    def compute_scores(self, source_feat, source_struct, lcg_feat, lcg_struct, cls_feat=None):
        raise NotImplementedError

    # ---------------- forward (baseline 과 동일한 in/out) ----------------
    def forward(self, source_struct, source_feat, lcg_struct, lcg_feat, batch, cls_feat=None):
        """
        Args:
            source_struct: [B, N, N]   (cos/xattn 에서는 미사용)
            source_feat:   [B, N, D]
            lcg_struct:    [M, K, K]   (cos/xattn 에서는 미사용)
            lcg_feat:      [M, K, D]
            cls_feat:      [B, D] or None  (pooled-CLS ablation 용)

        Returns:
            Fy_res_batch:     [B, M, K, D]
            lcg_struct_batch: [B, M, K, K]
            pi:               [B, M]
            reg_loss:         scalar
        """
        B, N, D = source_feat.shape
        M, K, _ = lcg_feat.shape

        # Step 1: LCG 를 배치 차원으로 복제 (baseline 과 동일)
        lcg_feat_batch = lcg_feat.unsqueeze(0).expand(B, M, K, D)
        lcg_struct_batch = lcg_struct.unsqueeze(0).expand(B, M, K, K)

        self.log_step += 1
        cls_feat = self._resolve_cls(cls_feat, source_feat)

        # Step 2: alignment score (여기만 모드별로 다름) + wall-clock
        # 측정은 training forward 에서만 (backward 제외, 세 방식 일관 적용)
        _timed = self.training and self.timer.active
        _ctx = self.timer.measure() if _timed else contextlib.nullcontext()
        with _ctx:
            scores = self.compute_scores(
                source_feat=source_feat,
                source_struct=source_struct,
                lcg_feat=lcg_feat,
                lcg_struct=lcg_struct,
                cls_feat=cls_feat,
            )

        if scores.shape != (B, M):
            raise RuntimeError(
                f"[{self.COORD_MODE}] compute_scores must return [B, M]={(B, M)}, got {tuple(scores.shape)}"
            )

        # Step 3: coordinate
        pi = torch.softmax(scores / self.coord_tau, dim=1)

        # Step 4: routing regularizer (baseline 과 동일)
        reg_loss, H_p, H_s = self._routing_reg(pi)
        self._last_H_p = float(H_p.detach())
        self._last_H_s = float(H_s.detach())

        self.last_pi = pi.detach()
        self.last_plan = None

        self._log_routing(pi, scores, batch)

        return lcg_feat_batch, lcg_struct_batch, pi, reg_loss

    # ---------------- logging ----------------
    def _log_routing(self, pi, scores, batch):
        if int(self.log_step) % self.log_interval != 0:
            return
        with torch.no_grad():
            src_idx = batch.get("src_idx", -1) if isinstance(batch, dict) else -1
            if isinstance(src_idx, torch.Tensor):
                src_idx = src_idx.item()
            M = pi.shape[1]
            pi_np = pi.detach().float().cpu().numpy()
            top1 = pi_np.max(axis=1)
            argmax_counts = torch.bincount(
                pi.argmax(dim=1).cpu(), minlength=M).numpy()
            s = scores.detach().float()
            per_range = (s.max(dim=1).values - s.min(dim=1).values)
            self.logger.info(
                f"[ROUTING/{self.COORD_MODE}] Step {int(self.log_step)} | Epoch {self.current_epoch} | "
                f"src={src_idx} | tau={self.coord_tau:.4f} | "
                f"H_s={self._last_H_s:.3f} H_p={self._last_H_p:.3f} | "
                f"top1={top1.mean():.3f}({top1.min():.3f}~{top1.max():.3f}) | "
                f"score gap mean={per_range.mean().item():.6f} | argmax={argmax_counts}"
            )
            if self.timer.enabled and self.timer.last_ms == self.timer.last_ms:
                self.logger.info(
                    f"   >>> [ALIGN-TIME/{self.COORD_MODE}] last={self.timer.last_ms:.3f} ms/batch "
                    f"(window={self.timer._phase}, collected={len(self.timer._samples)}/{self.timer.n_measure})"
                )

    # ---------------- 리포팅 유틸 ----------------
    def alignment_parameters(self):
        """이 모듈이 갖는 alignment 파라미터 (fgw/cos 는 빈 리스트)."""
        return list(self.named_parameters())

    def param_report(self):
        ps = self.alignment_parameters()
        return {
            "coord_mode": self.COORD_MODE,
            "uses_structure": bool(self.USES_STRUCTURE),
            "align_params_total": int(sum(p.numel() for _, p in ps)),
            "align_params_trainable": int(sum(p.numel() for _, p in ps if p.requires_grad)),
            "align_param_shapes": {n: tuple(p.shape) for n, p in ps},
        }

    def timing_report(self):
        rep = self.timer.report()
        rep["coord_mode"] = self.COORD_MODE
        return rep

    def rearm_timer(self, phase: str):
        self.timer.rearm(phase)

    def reset_timing(self):
        self.timer.reset()

    def freeze_alignment(self):
        """source 학습 후 target 에서 alignment 파라미터 고정."""
        n = 0
        for _, p in self.alignment_parameters():
            p.requires_grad = False
            n += p.numel()
        self.logger.info(f"[{self.COORD_MODE}] alignment params frozen: {n}")
        return n


# =====================================================================
# Ablation 1) Pooled CLS cosine
#   c = softmax( cos(z^(m), ḡ_k) / τ ),  ḡ_k = mean_T( F̄_k )
#   → A^(m), Ā_k 미사용. 학습 파라미터 0개.
# =====================================================================
class PooledCosineCoordQuantizer(_BaseCoordQuantizer):

    COORD_MODE = "cos"
    USES_STRUCTURE = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._warned_no_cls = False
        # 'cls' : Model 의 x_basis[:,0,:] 사용 (권장, 스펙)
        # 'mean': instance 노드 mean pooling (cls_feat 미전달 시 fallback)
        self.instance_pool = str(getattr(self.args, "coord_cos_instance", "cls"))
        self.proto_pool = str(getattr(self.args, "coord_cos_proto", "mean"))

    def compute_scores(self, source_feat, source_struct, lcg_feat, lcg_struct, cls_feat=None):
        # ---- instance side: 이미 계산된 CLS 임베딩 z^(m) ∈ R^d ----
        if self.instance_pool == "cls" and cls_feat is not None:
            z = cls_feat
        else:
            if self.instance_pool == "cls" and not self._warned_no_cls:
                self.logger.warning(
                    "[cos] cls_feat 이 전달되지 않아 source_feat mean-pooling 으로 대체합니다. "
                    "TabularFLM_S_.py 의 graph_quantizer 호출에 cls_feat=self.x_basis[:, 0, :] 를 추가하세요."
                )
                self._warned_no_cls = True
            z = source_feat.mean(dim=1)  # [B, D]

        # ---- prototype side: F̄_k ∈ R^{T×d} → mean pooling → ḡ_k ∈ R^d ----
        if self.proto_pool == "mean":
            g = lcg_feat.mean(dim=1)     # [M, D]
        elif self.proto_pool == "max":
            g = lcg_feat.max(dim=1).values
        else:
            raise ValueError(f"unknown coord_cos_proto: {self.proto_pool}")

        z_n = F.normalize(z.float(), dim=-1)
        g_n = F.normalize(g.float(), dim=-1)
        return (z_n @ g_n.t()).to(source_feat.dtype)  # [B, M] cosine similarity


# =====================================================================
# Ablation 2) Cross-attention
#   S_k = F̄_k W_Q (F̃^(m) W_K)^T / sqrt(d)      -> [T, N]
#   α   = softmax(S_k)  over instance node axis N
#   H_k = α (F̃^(m) W_V)                        -> [T, d]
#   s_k = linear( pool_T(H_k) )                 -> scalar
#   c   = softmax(s / τ)
#   → 가변 노드 수는 attention 이 흡수. 노드 속성만 사용, 쌍별 구조 미사용.
#   → W_Q/W_K/W_V + scorer 는 추가 파라미터 (반드시 같이 리포트)
# =====================================================================
class CrossAttentionCoordQuantizer(_BaseCoordQuantizer):

    COORD_MODE = "xattn"
    USES_STRUCTURE = False

    def __init__(self, args, *a, **kw):
        super().__init__(args, *a, **kw)
        D = int(args.input_dim)
        dh = getattr(args, "coord_attn_dim", None)
        self.d_head = int(dh) if dh else D

        self.W_Q = nn.Linear(D, self.d_head, bias=False)
        self.W_K = nn.Linear(D, self.d_head, bias=False)
        self.W_V = nn.Linear(D, self.d_head, bias=False)
        self.score_proj = nn.Linear(self.d_head, 1, bias=True)

        for m in (self.W_Q, self.W_K, self.W_V):
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)

        self.attn_pool = str(getattr(args, "coord_attn_pool", "mean"))
        self.attn_dropout = nn.Dropout(float(getattr(args, "coord_attn_dropout", 0.0)))

        self.logger.info(
            f"[xattn] d_head={self.d_head} | alignment params={self.param_report()['align_params_total']}"
        )

    def compute_scores(self, source_feat, source_struct, lcg_feat, lcg_struct, cls_feat=None):
        # source_feat : F̃^(m)  [B, N, D]
        # lcg_feat    : F̄      [M, K(=T), D]
        Q = self.W_Q(lcg_feat)                              # [M, T, dh]
        Kx = self.W_K(source_feat)                          # [B, N, dh]
        Vx = self.W_V(source_feat)                          # [B, N, dh]

        # S_k = F̄_k W_Q (F̃^(m) W_K)^T / sqrt(d)
        S = torch.einsum("mth,bnh->bmtn", Q, Kx) / math.sqrt(self.d_head)  # [B, M, T, N]

        # α = softmax over instance node axis (가변 N 을 여기서 흡수)
        alpha = torch.softmax(S, dim=-1)
        alpha = self.attn_dropout(alpha)

        # H_k = α (F̃^(m) W_V)
        H = torch.einsum("bmtn,bnh->bmth", alpha, Vx)       # [B, M, T, dh]

        # T 축 pooling → linear → 스칼라 s_k
        if self.attn_pool == "mean":
            h = H.mean(dim=2)
        elif self.attn_pool == "max":
            h = H.max(dim=2).values
        elif self.attn_pool == "sum":
            h = H.sum(dim=2)
        else:
            raise ValueError(f"unknown coord_attn_pool: {self.attn_pool}")

        return self.score_proj(h).squeeze(-1)               # [B, M]


# =====================================================================
# Baseline(fgw) + wall-clock: LCG.py 수정 없이 timing 만 덧입힌 subclass
# =====================================================================
if _HAS_FGW:

    class TimedGraphQuantizer(GraphQuantizer):
        """기존 FGW GraphQuantizer 와 수치적으로 완전히 동일.
        compute_fgw 구간만 AlignTimer 로 감싼다.
        cls_feat kwarg 는 받아서 무시 (세 모드 호출부를 통일하기 위함)."""

        COORD_MODE = "fgw"
        USES_STRUCTURE = True

        def __init__(self, args, *a, **kw):
            super().__init__(args, *a, **kw)
            self.args = args
            self.phase2_start_epoch = getattr(self, "phase2_start_epoch", None)
            self._parent_ref = None
            self.timer = WindowTimer(
                name="align/fgw",
                warmup=int(getattr(args, "coord_warmup_batches", 10)),
                n_measure=int(getattr(args, "coord_measure_batches", 50)),
                enabled=bool(getattr(args, "coord_time_profile", True)),
            )
            self._time_this_forward = False

        def compute_fgw(self, src_feat, src_str, tgt_feat, tgt_str):
            # cos/xattn 과 대응되는 구간: cost matrix 구성 + scaling + FGW solve 전체
            if self._time_this_forward and self.timer.active:
                with self.timer.measure():
                    return super().compute_fgw(src_feat, src_str, tgt_feat, tgt_str)
            return super().compute_fgw(src_feat, src_str, tgt_feat, tgt_str)

        def forward(self, source_struct, source_feat, lcg_struct, lcg_feat, batch, cls_feat=None):
            # training forward 만 측정 (backward 제외) — 세 방식 일관
            self._time_this_forward = bool(self.training)
            try:
                return super().forward(source_struct, source_feat,
                                       lcg_struct, lcg_feat, batch)
            finally:
                self._time_this_forward = False

        # --- 리포팅 인터페이스 통일 ---
        def attach_parent(self, model):  # fgw 는 CLS 불필요 (인터페이스만 맞춤)
            return None

        alignment_parameters = _BaseCoordQuantizer.alignment_parameters
        param_report = _BaseCoordQuantizer.param_report
        timing_report = _BaseCoordQuantizer.timing_report
        rearm_timer = _BaseCoordQuantizer.rearm_timer
        reset_timing = _BaseCoordQuantizer.reset_timing
        freeze_alignment = _BaseCoordQuantizer.freeze_alignment

else:  # pragma: no cover
    class TimedGraphQuantizer:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError(
                f"models.LCG import 실패로 fgw 모드를 쓸 수 없습니다: {_FGW_IMPORT_ERROR}"
            )


# =====================================================================
# Factory / 리포팅 헬퍼
# =====================================================================
_ALIASES = {
    "fgw": "fgw", "baseline": "fgw", "ours": "fgw",
    "cos": "cos", "cosine": "cos", "pooled_cos": "cos", "pooled_cls": "cos",
    "xattn": "xattn", "crossattn": "xattn", "cross_attention": "xattn",
}


def normalize_coord_mode(mode) -> str:
    key = str(mode).lower().strip()
    if key not in _ALIASES:
        raise ValueError(f"unknown coord_mode='{mode}'. choose from {COORD_MODES}")
    return _ALIASES[key]


def build_quantizer(args, **kwargs):
    """args.coord_mode 에 따라 quantizer 를 만든다 (기본 'fgw' = 기존과 동일)."""
    mode = normalize_coord_mode(getattr(args, "coord_mode", "fgw"))
    if mode == "fgw":
        q = TimedGraphQuantizer(args, **kwargs)
    elif mode == "cos":
        q = PooledCosineCoordQuantizer(args, **kwargs)
    else:
        q = CrossAttentionCoordQuantizer(args, **kwargs)
    logger.info(f"[build_quantizer] coord_mode={mode} | {q.param_report()}")
    return q


def coord_param_report(model):
    """alignment 파라미터 수 + 전체 모델 파라미터 수를 같이 반환."""
    q = getattr(model, "graph_quantizer", None)
    if q is None:
        return {}
    if hasattr(q, "param_report"):
        rep = dict(q.param_report())
    else:  # plain GraphQuantizer
        ps = list(q.named_parameters())
        rep = {
            "coord_mode": "fgw",
            "uses_structure": True,
            "align_params_total": int(sum(p.numel() for _, p in ps)),
            "align_params_trainable": int(sum(p.numel() for _, p in ps if p.requires_grad)),
            "align_param_shapes": {n: tuple(p.shape) for n, p in ps},
        }
    rep["model_params_total"] = int(sum(p.numel() for p in model.parameters()))
    rep["model_params_trainable"] = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad))
    return rep


def assert_alignment_frozen(model, strict: bool = True):
    """target adaptation 시 W_Q/W_K/W_V 가 실제로 고정됐는지 검증."""
    q = getattr(model, "graph_quantizer", None)
    if q is None:
        return True
    live = [n for n, p in q.named_parameters() if p.requires_grad]
    if live:
        msg = f"[assert_alignment_frozen] 아직 학습 중인 alignment param: {live}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False
    logger.info("[assert_alignment_frozen] OK — alignment params 전부 frozen")
    return True


def summarize_run(model, elapsed_sec=None, prefix="coord"):
    """wandb / json 에 그대로 넣을 수 있는 flat dict."""
    q = getattr(model, "graph_quantizer", None)
    out = {}
    for k, v in coord_param_report(model).items():
        if isinstance(v, (int, float, str, bool)):
            out[f"{prefix}/{k}"] = v
    if q is not None and hasattr(q, "timing_report"):
        rep = q.timing_report()
        out[f"{prefix}/coord_mode"] = rep.get("coord_mode", "?")
        for phase, w in (rep.get("windows") or {}).items():
            for k, v in w.items():
                if isinstance(v, (int, float)):
                    out[f"{prefix}/align_{phase}/{k}"] = v
    if elapsed_sec is not None:
        out[f"{prefix}/total_train_sec"] = float(elapsed_sec)
    return out


# =====================================================================
# shape smoke test:  python -m models.LCG_ablation
# =====================================================================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    B, N, D, M, K = 8, 13, 256, 12, 12
    a = argparse.Namespace(
        input_dim=D, n_graphs=M, n_nodes=K, struct_hidden_dim=192,
        soft_tau=0.01, tau=0.5, reg=0.01, vq_beta=0.3, entropy_reg=0.01,
        hs_reg=0.1, alpha=0.7, coord_time_profile=True, feat_distance='cosine',
        coord_warmup_batches=1, coord_measure_batches=3,
    )
    src_feat = torch.randn(B, N, D)
    src_str = torch.rand(B, N, N)
    lcg_feat = torch.randn(M, K, D)
    lcg_str = torch.rand(M, K, K)
    cls = torch.randn(B, D)
    batch = {"src_idx": 0}

    modes = ["cos", "xattn"] + (["fgw"] if _HAS_FGW else [])
    for mode in modes:
        a.coord_mode = mode
        q = build_quantizer(a)
        q.train()
        q.rearm_timer("smoke")
        for _ in range(6):
            fy, ls, pi, reg = q(src_str, src_feat, lcg_str, lcg_feat, batch, cls_feat=cls)
        print(f"[{mode}] Fy={tuple(fy.shape)} struct={tuple(ls.shape)} "
              f"pi={tuple(pi.shape)} sum={pi.sum(1).mean():.4f} reg={float(reg):.4f}")
        print(f"[{mode}] param_report={q.param_report()}")
        print(f"[{mode}] timing={q.timing_report()}")

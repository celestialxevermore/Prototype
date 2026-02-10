# Adapted from POT (Python Optimal Transport), MIT License
# https://github.com/PythonOT/POT
#
# Semi-relaxed Fused Gromov-Wasserstein batch solver
# Based on: ot.batch._utils.bregman_log_projection_batch
#           ot.batch._quadratic.solve_gromov_batch
#
# Key modification: target marginal constraint (T^T 1 = b) is removed.
# Only source marginal is enforced (T 1 = a).
# This allows LCG nodes to receive variable mass → more discriminative distances.
#
# Reference:
#   Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
#   "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
#   International Conference on Learning Representations (ICLR), 2022.

from ot.backend import get_backend
from ot.utils import OTResult

# Internal imports from POT batch modules
from ot.batch._utils import norm_batch, bop
from ot.batch._quadratic import (
    tensor_batch,
    tensor_product_batch,
    loss_quadratic_batch,
    detach_cost_tensor,
    transpose,
)
from ot.batch._linear import loss_linear_batch


# ============================================================
# 1) Semi-relaxed Bregman projection (v update removed)
# ============================================================
def semirelaxed_bregman_log_projection_batch(
    K, a=None, nx=None, max_iter=10000, tol=1e-5
):
    r"""
    Semi-relaxed Bregman projection for a batch of affinity matrices.

    Only the SOURCE marginal constraint is enforced:
        T 1 = a       (enforced via row scaling)
        T^T 1 = ???   (FREE — no column scaling)

    This is achieved by performing only the row (u) update in the
    Sinkhorn iterations, while keeping v fixed at 0.

    Mathematically:
        Standard:      u <- log(a) - LSE(K + v),  v <- log(b) - LSE(K^T + u)
        Semi-relaxed:  u <- log(a) - LSE(K + v),  v stays at 0

    Since v=0 → exp(v)=1 → no column normalization → target marginal is free.

    Parameters
    ----------
    K : (B, n, m) log-domain affinity matrix batch.
    a : (B, n) source distribution. If None, uniform.
    nx : backend (torch/numpy).
    max_iter : max Sinkhorn iterations.
    tol : convergence tolerance on source marginal error.

    Returns
    -------
    dict: 'T', 'log_T', 'potentials' (u, v), 'n_iters'
    """
    if nx is None:
        nx = get_backend(a, K)

    B, n, m = K.shape

    if a is None:
        a = nx.ones((B, n), type_as=K) / n

    u = nx.zeros((B, n), type_as=K)
    v = nx.zeros((B, m), type_as=K)  # initialized once, NEVER updated

    loga = nx.log(a)
    K = nx.detach(K)

    n_iters = 0
    for n_iters in range(max_iter):
        # ===== Row update only (source marginal enforcement) =====
        u = loga - nx.logsumexp(K + v[:, None, :], axis=2)
        # v = logb - nx.logsumexp(K + u[:, :, None], axis=1)  ← REMOVED!

        # Check convergence every 10 iterations
        if n_iters % 10 == 0:
            T = nx.exp(K + u[:, :, None] + v[:, None, :])
            marginal = nx.sum(T, axis=2)
            err = nx.max(norm_batch(marginal - a))
            if err < tol:
                break

    log_T = K + u[:, :, None] + v[:, None, :]
    T = nx.exp(log_T)

    return {
        "T": T,
        "log_T": log_T,
        "n_iters": n_iters,
        "potentials": (u, v),
    }


# ============================================================
# 2) Semi-relaxed solve_gromov_batch
# ============================================================
def solve_semirelaxed_gromov_batch(
    C1,
    C2,
    reg=1e-2,
    a=None,
    loss="sqeuclidean",
    symmetric=None,
    M=None,
    alpha=None,
    T_init=None,
    max_iter=50,
    tol=1e-5,
    max_iter_inner=50,
    tol_inner=1e-5,
    logits=None,
):
    r"""
    Semi-relaxed batch (Fused) Gromov-Wasserstein solver.

    Solves a batch of semi-relaxed (F)GW problems:

        min_T  (1-α)<M,T> + α Σ L(C1_ik, C2_jl) T_ij T_kl
        s.t.   T 1 = a       (source marginal — enforced)
               T^T 1 = ???   (target marginal — FREE)
               T ≥ 0

    Based on solve_gromov_batch with two modifications:
    1) Inner Bregman projection → semirelaxed (no column normalization)
    2) tensor_product uses recompute_const=True (target marginal changes)

    Parameters
    ----------
    C1 : (B, n, n) source structure matrices
    C2 : (B, m, m) target structure matrices (LCG)
    reg : float, entropic regularization (default 1e-2)
    a : (B, n) source marginal. If None, uniform.
    loss : str, 'sqeuclidean' or 'kl'
    symmetric : bool, whether C1/C2 are symmetric
    M : (B, n, m) feature cost matrix for Fused GW. None = pure GW.
    alpha : float, trade-off α*GW + (1-α)*W
    T_init : (B, n, m) initial transport plan
    max_iter : outer iterations (proximal gradient)
    tol : outer convergence tolerance
    max_iter_inner : inner Sinkhorn iterations
    tol_inner : inner convergence tolerance

    Returns
    -------
    OTResult with .plan, .value, .value_linear, .value_quad
    """

    nx = get_backend(a, M, C1, C2, T_init)
    B, n, m = C1.shape[0], C1.shape[1], C2.shape[1]

    if a is None:
        a = nx.ones((B, n), type_as=C1) / n

    # b_dummy: only used for tensor_batch initialization (constC).
    # Will be recomputed with actual T marginals via recompute_const=True.
    b_dummy = nx.ones((B, m), type_as=C2) / m

    if symmetric is None:
        symmetric = nx.allclose(
            C1, transpose(C1, nx=nx), atol=1e-10
        ) and nx.allclose(C2, transpose(C2, nx=nx), atol=1e-10)

    # ---------- Cost tensor (quadratic part) ---------- #
    if isinstance(loss, str):
        L = tensor_batch(
            a, b_dummy, C1, C2, symmetric=symmetric, nx=nx, loss=loss, logits=logits
        )
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    # ---------- Cost matrix (linear part) ---------- #
    if M is None and alpha is None:
        M = nx.zeros((B, n, m), type_as=C1)
        alpha = 1.0
    elif M is not None and alpha is None:
        raise ValueError("alpha must be provided when M is given")
    elif M is None and alpha is not None:
        raise ValueError("M must be provided when alpha is given")
    else:
        assert M.shape == (B, n, m), f"M shape {M.shape} != ({B}, {n}, {m})"
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"

    # ---------- Initialize T ---------- #
    if T_init is None:
        # Source marginal respected: T_init[i,j] = a[i] / m
        T_init = a[:, :, None] * nx.ones((B, 1, m), type_as=C1) / m

    T = T_init
    log_T = nx.log(T + 1e-15)

    # ---------- Proximal Gradient Descent ---------- #
    iter_count = 0
    for iter_count in range(max_iter):
        T_prev = T

        # Gradient of quadratic term
        # KEY DIFFERENCE: recompute_const=True because target marginal is FREE!
        # In standard FGW, constC is fixed (both marginals enforced).
        # In semi-relaxed, target marginal = T^T 1 changes each iteration.
        Mk = (1 - alpha) * M + 2 * alpha * tensor_product_batch(
            L, T, nx=nx, recompute_const=True, symmetric=symmetric
        )

        # Proximal step: K = -gradient/reg + log(T_prev)
        K = -Mk / reg + log_T

        # Inner solver: semi-relaxed Sinkhorn (only row constraint)
        out = semirelaxed_bregman_log_projection_batch(
            K, a, nx=nx, max_iter=max_iter_inner, tol=tol_inner
        )
        T = out["T"]
        log_T = out["log_T"]

        # Convergence check
        max_err = nx.max(nx.mean(nx.abs(T_prev - T), axis=(1, 2)))
        if max_err < tol:
            break

    # ---------- Compute final values (detached) ---------- #
    T = nx.detach(T)
    M_det = nx.detach(M)
    L_det = detach_cost_tensor(L, nx=nx)

    value_linear = loss_linear_batch(M_det, T, nx=nx)
    value_quadratic = loss_quadratic_batch(
        L_det, T, nx=nx, recompute_const=True, symmetric=symmetric
    )
    value = (1 - alpha) * value_linear + alpha * value_quadratic

    return OTResult(
        value=value,
        value_linear=value_linear * (1 - alpha),
        value_quad=value_quadratic * alpha,
        plan=T,
        backend=nx,
        potentials=out["potentials"],
        log={"n_iter": iter_count},
    )

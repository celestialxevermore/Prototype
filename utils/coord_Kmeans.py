# coord_kmeans.py
import numpy as np
import torch
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
logger = logging.getLogger(__name__)

@torch.no_grad()
def _collect_coordinates(model, loaders, device, max_batches=0, max_points=0):
    """
    모델의 get_coordinates(batch)를 이용해 좌표 C:[N,K]를 수집.
    (요청대로: 멀티소스를 단일 공간에 모아 클러스터링)
    """
    model.eval()
    buf = []
    for ld in loaders:
        for b_idx, batch in enumerate(ld):
            c = model.get_coordinates(batch).detach().cpu().numpy()  # [B,K]
            buf.append(c)
            if max_batches and (b_idx + 1) >= max_batches:
                break
    if not buf:
        raise RuntimeError("No coordinates collected.")
    C = np.concatenate(buf, axis=0).astype(np.float32)
    if max_points and len(C) > max_points:
        idx = np.random.choice(len(C), size=max_points, replace=False)
        C = C[idx]
    return C  # [N,K]

def _fit_kmeans(C, k, n_init='auto', max_iter=300, random_state=42):
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state, verbose=0)
    labels = km.fit_predict(C)
    return km, labels

def _silhouette(C, labels, max_eval=50000):
    n = len(C)
    if n > max_eval:
        idx = np.random.choice(n, size=max_eval, replace=False)
        C_eval = C[idx]
        labels_eval = labels[idx]
    else:
        C_eval = C
        labels_eval = labels
    return silhouette_score(C_eval, labels_eval, metric='euclidean')





@torch.no_grad()
def compute_coordinate_centroids_auto(
    model, loaders, device,
    k_min=2, k_max=16,
    metric='euclidean',
    max_batches=0,
    max_points=0,
    silhouette_sample=50000,
    random_state=42
):
    """
    멀티소스 전체를 하나의 공간으로 모아 KMeans.
    k ∈ [k_min, k_max] 중 silhouette가 최대인 k* 선택.
    return: (centroids:[k*,K] torch.FloatTensor, best_k:int, scores:dict)
    """
    assert metric == 'euclidean', "Only euclidean is supported."
    C = _collect_coordinates(model, loaders, device, max_batches=max_batches, max_points=max_points)

    best_k, best_score, best_centers = None, -1e9, None
    scores = {}
    for k in range(max(2, k_min), max(2, k_max) + 1):
        km, labels = _fit_kmeans(C, k, random_state=random_state)
        try:
            score = _silhouette(C, labels, max_eval=silhouette_sample)
        except Exception as e:
            logger.warning(f"[coord] silhouette failed at k={k}: {e}")
            score = float('-inf')
        scores[k] = float(score)
        if score > best_score:
            best_score  = score
            best_k      = k
            best_centers = km.cluster_centers_.astype(np.float32)

    if best_centers is None:
        logger.warning("[coord] silhouette selection failed; fallback to k_min")
        km = KMeans(n_clusters=max(2, k_min), random_state=random_state, n_init='auto', max_iter=300).fit(C)
        best_centers = km.cluster_centers_.astype(np.float32)
        best_k       = max(2, k_min)
        best_score   = float('nan')

    logger.info(f"[coord] best_k={best_k} silhouette={best_score:.4f}")
    return torch.from_numpy(best_centers), int(best_k), scores
def _normalize_simplex(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return x / x.sum(dim=-1, keepdim=True)
    
@torch.no_grad()
def build_centroid_target(c: torch.Tensor,
                          centroids: torch.Tensor,
                          tau: float = 0.3,
                          mode: str = "soft") -> torch.Tensor:
    """
    c: [B, K]  (device: whatever the model is on)
    centroids: [K*, K]
    return q: [B, K*]
    """
    device = c.device
    c = c.to(device=device, dtype=torch.float32)
    centroids = centroids.to(device=device, dtype=torch.float32)

    # 거리 기반 할당
    dists = torch.cdist(c, centroids, p=2)  # [B, K*]

    if mode == "hard":
        idx = torch.argmin(dists, dim=1)               # [B]
        q = torch.zeros_like(dists)                    # [B, K*]
        q.scatter_(1, idx.unsqueeze(1), 1.0)
        return q
    else:
        # soft assignment
        return F.softmax(-dists / max(tau, 1e-6), dim=1)
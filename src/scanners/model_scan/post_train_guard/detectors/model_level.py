"""
Model-level detectors, ported from the benchmark (model_detectors.py: Spectral,
Activation Clustering; rpp_detector.py: RPP), with the two fixes kept (RPP uses
the L-inf norm; AC has the silhouette + relative-size gate).

The host pipeline hands a MODEL scanner a state_dict, so we reconstruct a forward
pass from it (a plain feed-forward MLP) and query it. The single torch contact is
reading tensors out of the state_dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import extract_xy


# --------------------------------------------------------------------------- #
#  Forward-pass reconstruction
# --------------------------------------------------------------------------- #
def _to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):  # torch.Tensor
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=float)


def reconstruct_linear_layers(
    state_dict: Dict[str, Any]
) -> Tuple[Optional[List[Tuple[np.ndarray, Optional[np.ndarray]]]], Optional[str]]:
    np_items = {k: _to_numpy(v) for k, v in state_dict.items()}
    for k, v in np_items.items():
        if v.ndim >= 3:
            return None, f"unsupported tensor (ndim={v.ndim}) '{k}' — not a plain MLP"
    weight_keys = [k for k in np_items
                   if np_items[k].ndim == 2 and (k == "weight" or k.endswith(".weight"))]
    if not weight_keys:
        return None, "no Linear (2D *.weight) tensors found"
    layers: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
    for wk in weight_keys:
        W = np_items[wk]
        b = np_items.get(wk[:-6] + "bias")
        if b is not None and (b.ndim != 1 or b.shape[0] != W.shape[0]):
            b = None
        layers.append((W, b))
    for i in range(1, len(layers)):
        if layers[i][0].shape[1] != layers[i - 1][0].shape[0]:
            return None, "Linear layers do not chain — not a simple sequential MLP"
    return layers, None


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _proba(z: np.ndarray) -> np.ndarray:
    """Вероятности из логитов. Один выходной нейрон (бинарка, sklearn/torch с
    1-logit) -> сигмоида -> 2 колонки [1-p, p]; иначе softmax. Без этого softmax
    по одному логиту даёт вырожденные [1.0] (ломало RPP и model_diff на бинарных
    реконструированных MLP)."""
    if z.ndim == 2 and z.shape[1] == 1:
        p = 1.0 / (1.0 + np.exp(-z))
        return np.hstack([1.0 - p, p])
    return _softmax(z)


def mlp_forward(layers, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(representation = last hidden activation, proba). ReLU assumed."""
    h = X.astype(float)
    logits = h
    for i, (W, b) in enumerate(layers):
        z = h @ W.T
        if b is not None:
            z = z + b
        if i < len(layers) - 1:
            h = np.maximum(z, 0.0)
        else:
            logits = z
    repr_ = h if len(layers) >= 2 else logits
    return repr_, _proba(logits)


def prepare_model(state_dict, dataset) -> Tuple[list, np.ndarray, np.ndarray, str]:
    """Общий вход для всех model-level адаптеров: достаёт X/y из датасета и
    восстанавливает MLP из state_dict. Бросает ValueError с причиной (адаптер
    переведёт это в SKIPPED). Одно-слойные модели отклоняются (нет скрытого
    представления для Spectral/AC)."""
    if not state_dict:
        raise ValueError("model not loaded (no state_dict)")
    X, y, label_col = extract_xy(dataset)
    layers, reason = reconstruct_linear_layers(state_dict)
    if layers is None:
        raise ValueError(reason)
    in_features = int(layers[0][0].shape[1])
    if X.shape[1] != in_features:
        raise ValueError(f"feature count {X.shape[1]} != model input width {in_features}")
    if len(layers) < 2:
        raise ValueError("single linear layer — no hidden representation for Spectral/AC")
    return layers, X, y, label_col


# --------------------------------------------------------------------------- #
#  Spectral Signatures (Tran et al., 2018)
# --------------------------------------------------------------------------- #
def spectral_scores(repr_: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(repr_))
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < 5:
            continue
        R = repr_[idx]
        R_c = R - R.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(R_c, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        scores[idx] = (R_c @ Vt[0]) ** 2
    return scores


# --------------------------------------------------------------------------- #
#  Activation Clustering (Chen et al., 2018) + silhouette / relative-size gate
# --------------------------------------------------------------------------- #
def activation_clustering(
    repr_: np.ndarray,
    labels: np.ndarray,
    n_components: int = 10,
    seed: int = 42,
    sil_threshold: float = 0.10,
    max_poison_frac: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    flags = np.zeros(len(repr_), dtype=int)
    scores = np.zeros(len(repr_))
    max_minority_sil = -1.0
    n_gated = 0
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < 10:
            continue
        R = repr_[idx]
        k = min(n_components, R.shape[1], len(idx) - 1)
        R_red = PCA(n_components=k, random_state=seed).fit_transform(R) if k >= 2 else R
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(R_red)
        lab = km.labels_
        c0, c1 = int((lab == 0).sum()), int((lab == 1).sum())
        poison_cluster = 0 if c0 < c1 else 1
        scores[idx] = np.linalg.norm(R_red - km.cluster_centers_[1 - poison_cluster], axis=1)
        smaller_frac = min(c0, c1) / len(idx)
        try:
            sil = (silhouette_score(R_red, lab, sample_size=min(2000, len(idx)),
                                    random_state=seed) if len(np.unique(lab)) == 2 else -1.0)
        except Exception:
            sil = -1.0
        if smaller_frac <= max_poison_frac:
            max_minority_sil = max(max_minority_sil, sil)
        if sil >= sil_threshold and smaller_frac <= max_poison_frac:
            flags[idx] = (lab == poison_cluster).astype(int)
            n_gated += 1
    meta = {
        "max_minority_silhouette": round(float(max_minority_sil), 4),
        "n_gated_classes": int(n_gated),
        "gated_fraction": round(float(flags.mean()), 4),
    }
    return flags, scores, meta


# --------------------------------------------------------------------------- #
#  RPP (Lin et al., 2026) — L-inf prediction stability under Gaussian noise
# --------------------------------------------------------------------------- #
def rpp_scores(layers, X: np.ndarray, n_perturb: int = 20, sigma: float = 0.3,
               seed: int = 0) -> np.ndarray:
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1e-9
    _, p0 = mlp_forward(layers, X)
    rng = np.random.RandomState(seed)
    pert = np.zeros(len(X))
    for _ in range(n_perturb):
        _, pk = mlp_forward(layers, X + rng.normal(0.0, 1.0, X.shape) * (sigma * std))
        pert += np.abs(pk - p0).max(axis=1)  # L-inf, as in the paper
    return -(pert / n_perturb)  # stable (low pert) -> high suspicion

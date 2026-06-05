"""
RESERVED detector logic — parked for future integration into other modules
(e.g. a data-level / pre-train guard). NOT wired into any active pipeline.

Ported unchanged from the original ensemble:
  * charset_scan   — homoglyph (mixed Latin+Cyrillic tokens)
  * trigger_scan   — backdoor trigger tokens (reference-free)
  * securelearn_scores — per-class robust feature-outlier distance
  * knn_consensus_scores — kNN label-consensus (label-flip signal)
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[A-Za-z0-9\u0400-\u04FF']+")


def charset_scan(texts: List[str]) -> Tuple[int, Dict[str, int]]:
    """Homoglyph: слова со СМЕШАННЫМ шрифтом (латиница+кириллица в одном токене)."""
    rows_hit = 0
    examples: Dict[str, int] = {}
    for t in texts:
        hit = False
        for tok in _TOKEN.findall(t):
            if _LAT.search(tok) and _CYR.search(tok):
                hit = True
                if len(examples) < 10:
                    examples[tok] = examples.get(tok, 0) + 1
        if hit:
            rows_hit += 1
    return rows_hit, examples


def trigger_scan(texts: List[str], min_df_frac: float = 0.005) -> Tuple[int, List[str]]:
    """Backdoor-триггер (reference-free): токены (буквы+цифры, длина >= 6), повторяющиеся во многих строках."""
    n = len(texts)
    toks_per_row = [set(w.lower() for w in _TOKEN.findall(t)) for t in texts]
    df_count: Dict[str, int] = {}
    for toks in toks_per_row:
        for w in toks:
            df_count[w] = df_count.get(w, 0) + 1
    thr = max(5, min_df_frac * n)

    def suspicious(w: str) -> bool:
        has_digit = any(ch.isdigit() for ch in w)
        has_alpha = any(ch.isalpha() and ch.isascii() for ch in w)
        return len(w) >= 6 and has_digit and has_alpha

    trigger_tokens = sorted(w for w, c in df_count.items() if c >= thr and suspicious(w))
    rows_hit = sum(1 for toks in toks_per_row if toks & set(trigger_tokens))
    return rows_hit, trigger_tokens


def securelearn_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SecureLearn: per-class робастное расстояние точки до центра своего класса."""
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    scores = np.zeros(len(Xs))
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 5:
            continue
        Xc = Xs[idx]
        center = np.median(Xc, axis=0)
        mad = np.median(np.abs(Xc - center), axis=0) * 1.4826
        mad[mad < 1e-6] = 1e-6
        scores[idx] = np.sqrt(np.sum(((Xc - center) / mad) ** 2, axis=1))
    return scores


def knn_consensus_scores(X: np.ndarray, y: np.ndarray, k: int = 15) -> np.ndarray:
    """kNN-консенсус: score = 1 - P(заявленная метка) по соседям (ловит label flip)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2 or len(X) < k + 1:
        return np.zeros(len(X))
    Xs = StandardScaler().fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(Xs) - 1)).fit(Xs, y)
    proba = knn.predict_proba(Xs)
    cls = {c: j for j, c in enumerate(knn.classes_)}
    s = np.ones(len(Xs))
    for i, lab in enumerate(y):
        if lab in cls:
            s[i] = 1.0 - proba[i, cls[lab]]
    return s

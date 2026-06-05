"""
Shared helpers used by the post-train detectors (label/feature extraction,
robust-outlier flagging, text-column detection). The data-level *detectors*
themselves (SecureLearn, kNN, charset, trigger) now live in the top-level
`reserved/` package, pending integration into a future data-level module.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

LABEL_CANDIDATES = ("label", "target", "y", "class", "Class")


def find_label_column(df) -> str | None:
    return next((c for c in LABEL_CANDIDATES if c in df.columns), None)


def extract_xy(df) -> Tuple[np.ndarray, np.ndarray, str]:
    """(числовые признаки X, метки y как int, имя колонки-метки). Медианная
    импутация NaN. Бросает ValueError с понятной причиной."""
    import pandas as pd

    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError("no label column (need label/target/y/class)")
    y = pd.factorize(df[label_col], sort=True)[0].astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("fewer than 2 classes in labels")
    feat = df.drop(columns=[label_col]).select_dtypes(include="number")
    if feat.shape[1] == 0:
        raise ValueError("no numeric feature columns")
    X = feat.to_numpy(dtype=float)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
    return X, y, label_col


def outlier_flags(scores: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Робастные выбросы: score > median + k*1.4826*MAD.

    ОСТОРОЖНО: зависит от ФОРМЫ распределения. У тяжёлого правого хвоста (Spectral
    = квадрат проекции, ~χ²) переметит ~10-15% даже на чистом; у ограниченного
    сверху скора (RPP = −perturbation, стабильность насыщается у нуля) недометит
    даже реальную закладку (см. fin_phrasebank d2: AUC=1.0, а flagged=0). Для
    model-level Spectral/RPP используйте top_fraction_flags (ранг-гейт)."""
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    if mad < 1e-9:
        return np.zeros(len(scores), dtype=int)
    return (scores > med + k * mad).astype(int)


def top_fraction_flags(scores: np.ndarray, frac: float) -> np.ndarray:
    """Самодостаточный РАНГ/КВАНТИЛЬ-гейт: помечает верхние `frac` строк по скору.
    НЕ требует чистых данных и НЕ зависит от формы распределения (в отличие от
    outlier_flags). Это ТРИАЖ: верхние ε кандидатов на ручной просмотр, а НЕ
    калиброванный вердикт «отравлено/чисто» — без эталона его дать нельзя
    (проверено: ни форма распределения скоров, ни консенсус детекторов не
    разделяют чистое и отравленное). Ценность model-level — РАНЖИРОВАНИЕ строк."""
    n = len(scores)
    if n == 0 or frac <= 0:
        return np.zeros(n, dtype=int)
    k = min(max(1, int(round(frac * n))), n)
    thr = np.partition(scores, n - k)[n - k]
    return (scores >= thr).astype(int)


def top_indices(scores: np.ndarray, n: int = 20) -> List[int]:
    return np.argsort(scores)[::-1][: min(n, len(scores))].astype(int).tolist()


def permutation_flag(stat_fn, reps, labels, K: int = 15, q: float = 95.0,
                     margin: float = 0.0, floor=None, seed: int = 1):
    """Самокалиброванный гейт под датасет (без эталона): сравнить реальную
    статистику с НУЛЁМ из перемешанных меток (stat_fn(reps, shuffled_labels)).
    Флаг, если real > перцентиль(null, q) + margin [и real >= floor, если задан].
    Перестановка ломает внутриклассовую структуру отравления, оставляя
    «естественную» как фон -> отделяет аномалию от природной структуры."""
    real = float(stat_fn(reps, labels))
    rng = np.random.RandomState(seed)
    lab = np.asarray(labels)
    null = np.array([float(stat_fn(reps, rng.permutation(lab))) for _ in range(K)])
    thr = float(np.percentile(null, q)) + margin
    flagged = (real > thr) and (floor is None or real >= floor)
    return bool(flagged), round(real, 3), round(thr, 3)


def cluster_gap_flag(scores, min_frac: float = 0.03, max_frac: float = 0.40,
                     sep: float = 1.2, seed: int = 0):
    """Self-contained гейт для 1-D скоров БЕЗ меток (RPP): 2-means, флаг если есть
    обособленный кластер-МЕНЬШИНСТВО на высокой стороне (доля в [min,max]) с
    разделением центров sep (в единицах within-std). Консервативно."""
    from sklearn.cluster import KMeans
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    if len(x) < 20:
        return False, 0.0, 0.0
    lab = KMeans(2, n_init=5, random_state=seed).fit_predict(x)
    m0, m1 = float(x[lab == 0].mean()), float(x[lab == 1].mean())
    hi = 0 if m0 > m1 else 1
    frac = float((lab == hi).mean())
    sd = float(np.sqrt((x[lab == hi].std() ** 2 + x[lab != hi].std() ** 2) / 2 + 1e-12))
    d = abs(m0 - m1) / sd
    return bool(min_frac <= frac <= max_frac and d >= sep), round(frac, 3), round(d, 3)


def bootstrap_upper_quantile(x: np.ndarray, q: float, n_boot: int = 200,
                             ci: float = 0.95, seed: int = 0) -> float:
    """Верхняя CI-граница q-квантиля по бутстрапу. Для МАЛЫХ чистых выборок это
    расширяет порог консервативно (меньше ложных срабатываний), для больших —
    сходится к обычной квантили. Используется калибровкой по чистому сэмплу."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("inf")
    if len(x) < 10:                       # выборка крошечная — берём максимум (макс. консервативно)
        return float(np.max(x))
    rng = np.random.RandomState(seed)
    qs = np.array([np.quantile(x[rng.randint(0, len(x), len(x))], q) for _ in range(n_boot)])
    return float(np.quantile(qs, ci))


def text_columns(df, min_mean_len: float = 15.0) -> List[str]:
    """Свободно-текстовые колонки: не-числовые со средней длиной >= порога."""
    cols: List[str] = []
    for c in df.select_dtypes(exclude="number").columns:
        s = df[c].astype(str)
        if float(s.str.len().mean()) >= min_mean_len:
            cols.append(c)
    return cols


def combined_text(df, cols: List[str]) -> List[str]:
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s.str.cat(df[c].astype(str), sep=" ")
    return s.tolist()

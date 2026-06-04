"""
Data-level detectors + shared helpers, ported from the benchmark
(baselines.py: SecureLearn; nlp_detectors.py: charset/rare-token/kNN).

These are the validated detector *functions*; the thin scanner adapters in
src/scanners/<category>/ import from here and wrap the result in a ScanResult.
Adding a detector = add an adapter; removing one = delete the adapter file.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np

# Метки, которые ищем в датасете (для per-class детекторов)
LABEL_CANDIDATES = ("label", "target", "y", "class", "Class")


# --------------------------------------------------------------------------- #
#  Общие хелперы
# --------------------------------------------------------------------------- #
def find_label_column(df) -> str | None:
    return next((c for c in LABEL_CANDIDATES if c in df.columns), None)


def extract_xy(df) -> Tuple[np.ndarray, np.ndarray, str]:
    """(числовые признаки X, метки y как int, имя колонки-метки). Медианная
    импутация NaN. Бросает ValueError с понятной причиной, если не получается."""
    import pandas as pd  # noqa: F401

    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError("no label column (need label/target/y/class)")
    import pandas as pd
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
    """Робастные выбросы в распределении скоров: score > median + k*1.4826*MAD.
    Не требует фиксированной квантили/калибровки — у чистых данных хвост короткий,
    у отравленных появляется явная группа высоких скоров."""
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) * 1.4826
    if mad < 1e-9:
        return np.zeros(len(scores), dtype=int)
    return (scores > med + k * mad).astype(int)


def top_indices(scores: np.ndarray, n: int = 20) -> List[int]:
    return np.argsort(scores)[::-1][: min(n, len(scores))].astype(int).tolist()


# --------------------------------------------------------------------------- #
#  Tabular data-level
# --------------------------------------------------------------------------- #
def securelearn_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SecureLearn: per-class робастное расстояние точки до центра своего класса
    (median + MAD по признакам). Высокое расстояние = не похожа на свой класс
    (label flip / порча признаков). Самореференсно, без чистого эталона."""
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
    """kNN-консенсус: score = 1 - P(заявленная метка) по соседям. Ловит label
    flip (у перевёрнутой точки соседи другого класса)."""
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


# --------------------------------------------------------------------------- #
#  Text data-level (NLP)
# --------------------------------------------------------------------------- #
_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[A-Za-z0-9\u0400-\u04FF']+")


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
    """Backdoor-триггер (reference-free): «неприродные» токены (буквы+цифры,
    длина >= 6), повторяющиеся во многих строках."""
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

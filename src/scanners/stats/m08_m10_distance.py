"""
Детекторы на основе расстояний и соседства.

M8   Расстояние Махаланобиса + MinCovDet  — многомерные выбросы внутри класса
M9   KNN-согласованность меток            — детектор подмены меток (наиболее чувствительный)
M10  Обнаружение почти-дубликатов         — строки с одинаковыми признаками, но разными метками
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.core.factory import register_scanner
from src.core.features import FeatureBundle, reduce_to_n_components
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MAX_CLASSES = 50
_MAX_SAMPLE_PER_CLASS = 2000   # MinCovDet & LOF are O(n²) in feature space
_MAX_SAMPLE_GLOBAL = 5000      # for M9 / M10


def _bundle(ctx: ScanContext) -> Optional[FeatureBundle]:
    return ctx.metadata.get("features")


def _safe_pca(X: np.ndarray, n_cls: int) -> np.ndarray:
    """Снижает размерность до уровня, при котором MinCovDet работает корректно (n > 2p)."""
    max_p = max(2, n_cls // 3)
    return reduce_to_n_components(X, max_p) if X.shape[1] > max_p else X


def _sample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[np.random.default_rng(42).choice(len(arr), n, replace=False)]


def _clean(X: np.ndarray) -> np.ndarray:
    """Заменяет NaN/inf на 0, иначе дистанционные методы sklearn упадут."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# M8 — Mahalanobis + MinCovDet
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class MahalanobisScanner(BaseScanner):
    """
    M8: Робастное расстояние Махаланобиса через MinCovDet (Rousseeuw & Van Driessen, 1999).
    Ловит многомерные выбросы, которые Z-score пропускает — с учётом корреляций признаков.
    HAND_CHECK если в любом классе более 10% образцов выходят за χ²₀.₉₇₅.
    """
    name = "Stats: M8 Mahalanobis (MinCovDet)"
    category = ScannerCategory.STATS

    _OUTLIER_RATE_WARN = 0.10

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.covariance import MinCovDet
        from scipy.stats import chi2

        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})

        y = b.y
        classes = b.classes()
        if len(classes) > _MAX_CLASSES:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({len(classes)})"})

        anomalies: Dict[str, dict] = {}

        for cls in classes:
            X_c = _sample(_clean(b.X[y == cls]), _MAX_SAMPLE_PER_CLASS)
            n_c = len(X_c)
            if n_c < 20:
                continue
            X_c = _safe_pca(X_c, n_c)
            p = X_c.shape[1]
            if n_c < 2 * p + 2:
                continue  # MinCovDet requirement
            try:
                mcd = MinCovDet(support_fraction=0.75, random_state=42)
                mcd.fit(X_c)
                dist2 = mcd.mahalanobis(X_c)
                threshold = chi2.ppf(0.975, df=p)
                outlier_rate = float((dist2 > threshold).mean())
                if outlier_rate > self._OUTLIER_RATE_WARN:
                    anomalies[str(cls)] = {
                        "outlier_rate_pct": round(outlier_rate * 100, 2),
                        "n_samples": n_c,
                        "n_features_used": p,
                        "chi2_threshold": round(float(threshold), 2),
                    }
            except Exception as exc:
                continue

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "outlier_rate_warn_pct": self._OUTLIER_RATE_WARN * 100})


# ─────────────────────────────────────────────────────────────────────────────
# M9 — KNN label consistency  (most powerful label-flip detector)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class KNNConsistencyScanner(BaseScanner):
    """
    M9: KNN-согласованность меток (Biggio et al., 2011; Paudice et al., 2019).
    Несогласованность = доля k ближайших соседей с другой меткой.
    Метрика нормируется на импurity Джини — это убирает артефакт дисбаланса классов.
    FAILED если среднее_несогл / Gini ≥ 0.92 (метки неотличимы от случайных).
    """
    name = "Stats: M9 KNN label consistency"
    category = ScannerCategory.STATS

    _K = 5                       # k=5 из статей Biggio et al. и Paudice et al.
    _MAX_CLASSES_FOR_KNN = 20   # при > 20 классах (banking77) метрика неинформативна
    _RATIO_FAIL = 0.92           # метки почти не лучше случайных
    _RATIO_WARN = 0.80           # слабая кластеризация по меткам

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.neighbors import NearestNeighbors

        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})

        X_full = _clean(b.X)
        y_full = b.y
        n_total = len(X_full)

        n_classes = len(np.unique(y_full))
        if n_classes > self._MAX_CLASSES_FOR_KNN:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({n_classes}) — "
                                                 "KNN inconsistency is not informative"})

        if n_total > _MAX_SAMPLE_GLOBAL:
            idx = np.random.default_rng(42).choice(n_total, _MAX_SAMPLE_GLOBAL, replace=False)
            X_s, y_s = X_full[idx], y_full[idx]
        else:
            X_s, y_s = X_full, y_full

        k = min(self._K, len(X_s) - 1)
        if k < 1:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Too few samples"})

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_s)
        _, indices = nbrs.kneighbors(X_s)

        inc_arr = np.array([
            float((y_s[neighbours[1:]] != y_s[i]).mean())
            for i, neighbours in enumerate(indices)
        ])
        mean_inc = float(inc_arr.mean())

        # Импurity Джини = ожидаемая несогласованность при полностью случайных метках
        classes, counts = np.unique(y_s, return_counts=True)
        p = counts / counts.sum()
        gini = float(1.0 - np.sum(p ** 2))
        if gini < 1e-6:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Only one class present"})

        ratio = mean_inc / gini   # 0 = идеальная кластеризация, 1 = случайные метки

        if ratio >= self._RATIO_FAIL:
            status, passed = ScanStatus.FAILED, False
        elif ratio >= self._RATIO_WARN:
            status, passed = ScanStatus.HAND_CHECK, True
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=passed,
                          details={
                              "k_neighbours": k,
                              "n_samples_checked": len(X_s),
                              "mean_inconsistency": round(mean_inc, 4),
                              "gini_baseline": round(gini, 4),
                              "inconsistency_ratio": round(ratio, 4),
                              "fail_threshold_ratio": self._RATIO_FAIL,
                          })


# ─────────────────────────────────────────────────────────────────────────────
# M10 — Near-duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class NearDuplicateScanner(BaseScanner):
    """
    M10: Обнаружение почти-дубликатов с разными метками (Huang et al., 2011).
    Пары с расстоянием < 5-й перцентиль NN-дистанций и разными метками — признак атаки
    «скопировать пример и подменить метку».
    """
    name = "Stats: M10 near-duplicate detection"
    category = ScannerCategory.STATS

    _CONFLICT_RATE_WARN = 0.03   # >3 % HAND_CHECK
    _CONFLICT_RATE_FAIL = 0.08   # >8 % FAILED

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.neighbors import NearestNeighbors

        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})

        X_full, y_full = _clean(b.X), b.y
        n_total = len(X_full)

        if n_total > 2000:
            idx = np.random.default_rng(0).choice(n_total, 2000, replace=False)
            X_s, y_s = X_full[idx], y_full[idx]
        else:
            X_s, y_s = X_full, y_full

        n = len(X_s)
        if n < 10:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Too few samples"})

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_s)
        dists, indices = nbrs.kneighbors(X_s)

        nn_dists = dists[:, 1]           # расстояние до ближайшего другого образца
        theta = float(np.percentile(nn_dists, 5))

        conflict_count = 0
        for i in range(n):
            j = indices[i, 1]
            if nn_dists[i] <= theta and y_s[i] != y_s[j]:
                conflict_count += 1

        conflict_rate = conflict_count / n

        if conflict_rate >= self._CONFLICT_RATE_FAIL:
            status, passed = ScanStatus.FAILED, False
        elif conflict_rate >= self._CONFLICT_RATE_WARN:
            status, passed = ScanStatus.HAND_CHECK, True
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=passed,
                          details={
                              "n_samples_checked": n,
                              "distance_threshold": round(float(theta), 4),
                              "conflict_pairs": conflict_count,
                              "conflict_rate_pct": round(conflict_rate * 100, 3),
                              "warn_threshold_pct": self._CONFLICT_RATE_WARN * 100,
                              "fail_threshold_pct": self._CONFLICT_RATE_FAIL * 100,
                          })
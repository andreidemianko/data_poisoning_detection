"""
Плотностные и ансамблевые детекторы выбросов (по классам, без модели).

M11  Isolation Forest  — аномальность через глубину изоляции (Liu et al., ICDM 2008)
M12  LOF               — локальный коэффициент плотности (Breunig et al., SIGMOD 2000)
M13  DBSCAN            — шумовые точки как аномалии (Ester et al., KDD 1996)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.core.factory import register_scanner
from src.core.features import FeatureBundle, reduce_to_n_components
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MAX_CLASSES = 50
_MAX_SAMPLE_PER_CLASS = 2000


def _bundle(ctx: ScanContext) -> Optional[FeatureBundle]:
    return ctx.metadata.get("features")


def _sample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[np.random.default_rng(42).choice(len(arr), n, replace=False)]


def _clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # sklearn не переваривает NaN/inf


def _pca_if_needed(X: np.ndarray, max_p: int = 50) -> np.ndarray:
    return reduce_to_n_components(X, max_p) if X.shape[1] > max_p else X


# ─────────────────────────────────────────────────────────────────────────────
# M11 — Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class IsolationForestScanner(BaseScanner):
    """
    M11: Isolation Forest по каждому классу (Liu et al., 2008).
    contamination="auto" — порог на score=-0.5, из оригинальной статьи.
    Доля выбросов определяется данными, а не фиксируется в 10%.
    HAND_CHECK если любой класс превышает 20%.
    """
    name = "Stats: M11 Isolation Forest"
    category = ScannerCategory.STATS

    _WARN_RATE = 0.20   # >20 % outliers (data-driven threshold, not fixed contamination)

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.ensemble import IsolationForest

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
            X_c = _sample(_pca_if_needed(_clean(b.X[y == cls])), _MAX_SAMPLE_PER_CLASS)
            if len(X_c) < 20:
                continue
            # contamination="auto": порог на score=-0.5 (Liu et al. 2008)
            # доля выбросов определяется данными — без артефакта фиксированных 10%
            iso = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
            iso.fit(X_c)
            anomaly_rate = float((iso.predict(X_c) == -1).mean())
            if anomaly_rate > self._WARN_RATE:
                anomalies[str(cls)] = {
                    "anomaly_rate_pct": round(anomaly_rate * 100, 2),
                    "n_samples": len(X_c),
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "contamination": "auto",
                                   "warn_rate_pct": self._WARN_RATE * 100})


# ─────────────────────────────────────────────────────────────────────────────
# M12 — Local Outlier Factor
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class LOFScanner(BaseScanner):
    """
    M12: LOF по каждому классу (Breunig et al., SIGMOD 2000).
    Использует сырые LOF-оценки (не бинарные предсказания): LOF > 1.5 — порог из оригинальной статьи.
    HAND_CHECK если любой класс превышает 20%.
    """
    name = "Stats: M12 LOF"
    category = ScannerCategory.STATS

    _LOF_THRESHOLD = 1.5    # Breunig et al. (2000): LOF > 1 is anomalous; 1.5 is a practical cut
    _WARN_RATE = 0.20       # >20 % of class samples with LOF > 1.5 → suspicious

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.neighbors import LocalOutlierFactor

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
            X_c = _sample(_pca_if_needed(_clean(b.X[y == cls]), max_p=30), _MAX_SAMPLE_PER_CLASS)
            n_c = len(X_c)
            if n_c < 20:
                continue
            n_neighbors = min(20, n_c - 1)
            # novelty=False: оценка на обучающих данных; negative_outlier_factor_ → сырые LOF-оценки
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=-1)
            lof.fit(X_c)
            # -lof.negative_outlier_factor_ даёт положительные LOF: 1.0 = норма, > 1.5 = выброс
            lof_scores = -lof.negative_outlier_factor_
            anomaly_rate = float((lof_scores > self._LOF_THRESHOLD).mean())
            if anomaly_rate > self._WARN_RATE:
                anomalies[str(cls)] = {
                    "anomaly_rate_pct": round(anomaly_rate * 100, 2),
                    "n_samples": n_c,
                    "lof_threshold": self._LOF_THRESHOLD,
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "lof_threshold": self._LOF_THRESHOLD,
                                   "warn_rate_pct": self._WARN_RATE * 100})


# ─────────────────────────────────────────────────────────────────────────────
# M13 — DBSCAN noise points
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class DBSCANScanner(BaseScanner):
    """
    M13: DBSCAN по каждому классу — шумовые точки (label=-1) как кандидаты в аномалии.
    Epsilon выбирается автоматически по 90-му перцентилю k-NN дистанций.
    HAND_CHECK если в любом классе более 25% шумовых точек.
    """
    name = "Stats: M13 DBSCAN"
    category = ScannerCategory.STATS

    _WARN_NOISE = 0.25
    _MIN_SAMPLES = 5

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

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
            X_c = _sample(_pca_if_needed(b.X[y == cls], max_p=20), _MAX_SAMPLE_PER_CLASS)
            n_c = len(X_c)
            if n_c < 20:
                continue
            k = min(self._MIN_SAMPLES, n_c - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_c)
            dists, _ = nbrs.kneighbors(X_c)
            eps = float(np.percentile(dists[:, -1], 90))
            if eps < 1e-10:
                continue
            labels = DBSCAN(eps=eps, min_samples=k).fit_predict(X_c)
            noise_rate = float((labels == -1).mean())
            if noise_rate > self._WARN_NOISE:
                anomalies[str(cls)] = {
                    "noise_rate_pct": round(noise_rate * 100, 2),
                    "eps": round(eps, 4),
                    "n_samples": n_c,
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "warn_noise_rate_pct": self._WARN_NOISE * 100})
"""
Одномерные статистические детекторы (без модели).

M1  Modified Z-score (MAD)    — выбросы по каждому признаку внутри класса
M2  Моменты высших порядков    — скошенность и куртозис; двугорбость как след подмены меток
M3  Тест Колмогорова-Смирнова  — внутренняя согласованность класса (половинное разбиение)
M5  Jensen-Shannon Divergence  — расхождение распределений признаков между классами
M6  Расстояние Вассерштейна    — каждый класс против глобального распределения
M7  Критерий хи-квадрат        — связь категориальных признаков с меткой (по сырым данным)

M4 (PSI) не реализован — требует отдельного чистого референсного датасета.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, chi2_contingency

from src.core.factory import register_scanner
from src.core.features import FeatureBundle, find_label_column
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MAX_CLASSES = 50     # skip per-class loops for very many classes
_MAX_SAMPLE = 3000    # max samples per class for slow per-feature tests


def _bundle(ctx: ScanContext) -> Optional[FeatureBundle]:
    return ctx.metadata.get("features")


def _sample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[np.random.default_rng(42).choice(len(arr), n, replace=False)]


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Modified Z-score (MAD-based)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class ModifiedZScoreScanner(BaseScanner):
    """
    M1: Модифицированный Z-score на основе MAD (Iglewicz & Hoaglin, 1993).
    Ловит инъекцию выбросов и аномалии признаков внутри классов.
    Порог: |Mz| > 3.5 у более чем 5% образцов в классе.
    """
    name = "Stats: M1 modified z-score (MAD)"
    category = ScannerCategory.STATS

    _ZSCORE = 3.5
    _FAIL_RATE = 0.05   # >5% outliers per feature-class → anomaly

    def run(self, context: ScanContext) -> ScanResult:
        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})
        X = b.X_tabular()
        if X.shape[1] == 0:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No tabular features"})

        y, names = b.y, b.tabular_feature_names()
        classes = b.classes()
        if len(classes) > _MAX_CLASSES:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({len(classes)})"})

        anomalies: Dict[str, list] = {}
        for cls in classes:
            X_c = _sample(X[y == cls], _MAX_SAMPLE)
            if len(X_c) < 10:
                continue
            hits = []
            for j in range(X.shape[1]):
                col = X_c[:, j]
                med = np.median(col)
                mad = np.median(np.abs(col - med))
                if mad < 1e-10:
                    continue
                rate = float((0.6745 * np.abs(col - med) / mad > self._ZSCORE).mean())
                if rate >= self._FAIL_RATE:
                    hits.append({"feature": names[j], "outlier_rate_pct": round(rate * 100, 2)})
            if hits:
                anomalies[str(cls)] = hits[:10]

        # Always HAND_CHECK — heavy-tailed PCA/financial features trigger naturally;
        # hard outlier injection is more reliably caught by NumericSentinelValueScanner.
        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"zscore_threshold": self._ZSCORE,
                                   "fail_rate_pct": self._FAIL_RATE * 100,
                                   "anomalous_classes": anomalies})


# ─────────────────────────────────────────────────────────────────────────────
# M2 — High-order moments (Skewness & Kurtosis)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class HighOrderMomentsScanner(BaseScanner):
    """
    M2: Скошенность и куртозис по каждому признаку и классу (Barreno et al., 2010).
    Тяжёлый правый хвост (высокий куртозис) → инъекция выбросов.
    Плоское / двугорбое распределение (низкий куртозис) → загрязнение подменой меток.
    """
    name = "Stats: M2 high-order moments"
    category = ScannerCategory.STATS

    # Barreno et al. (2010): |скошенность| > 2, куртозис > 10
    _SKEW = 2.0
    _KURT_HIGH = 10.0   # тяжёлый хвост
    _KURT_LOW = -1.5    # двугорбое / платикуртическое распределение
    _FRAC_WARN = 0.20   # > 20% признаков с аномалиями в одном классе

    def run(self, context: ScanContext) -> ScanResult:
        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})
        X = b.X_tabular()
        if X.shape[1] == 0:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No tabular features"})

        y, names = b.y, b.tabular_feature_names()
        classes = b.classes()
        if len(classes) > _MAX_CLASSES:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({len(classes)})"})

        anomalies: Dict[str, dict] = {}
        for cls in classes:
            X_c = _sample(X[y == cls], _MAX_SAMPLE)
            if len(X_c) < 20:
                continue
            sk = stats.skew(X_c, axis=0)
            ku = stats.kurtosis(X_c, axis=0)
            mask = (np.abs(sk) > self._SKEW) | (ku > self._KURT_HIGH) | (ku < self._KURT_LOW)
            frac = float(mask.mean())
            if frac >= self._FRAC_WARN:
                anomalies[str(cls)] = {
                    "anomalous_features_pct": round(frac * 100, 1),
                    "examples": [{"feature": names[j],
                                  "skewness": round(float(sk[j]), 2),
                                  "kurtosis": round(float(ku[j]), 2)}
                                 for j in np.where(mask)[0][:5]],
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "thresholds": {"skewness": self._SKEW,
                                                  "kurtosis_high": self._KURT_HIGH,
                                                  "kurtosis_low": self._KURT_LOW}})


# ─────────────────────────────────────────────────────────────────────────────
# M3 — KS test: within-class internal consistency (half-split)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class KolmogorovSmirnovScanner(BaseScanner):
    """
    M3: KS-тест для проверки внутренней согласованности класса (половинное разбиение).
    Класс делится пополам случайно; если половины значительно расходятся по многим признакам —
    внутри класса смешаны два разных источника данных (загрязнение подменой меток).
    Порог: D_KS > 0.10 после поправки Бонферрони, более 20% признаков → HAND_CHECK.
    """
    name = "Stats: M3 KS within-class consistency"
    category = ScannerCategory.STATS

    _D_THRESHOLD = 0.10
    _FRAC_WARN = 0.20

    def run(self, context: ScanContext) -> ScanResult:
        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})
        X = b.X_tabular()
        if X.shape[1] == 0:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No tabular features"})

        y, names = b.y, b.tabular_feature_names()
        classes = b.classes()
        if len(classes) > _MAX_CLASSES:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({len(classes)})"})

        rng = np.random.default_rng(0)
        anomalies: Dict[str, dict] = {}
        alpha = 0.05 / max(X.shape[1] * len(classes), 1)  # Bonferroni

        for cls in classes:
            idx = np.where(y == cls)[0]
            if len(idx) < 30:
                continue
            rng.shuffle(idx)
            half = len(idx) // 2
            A, B = X[idx[:half]], X[idx[half:]]
            flagged = []
            for j in range(X.shape[1]):
                d, p = stats.ks_2samp(A[:, j], B[:, j])
                if d > self._D_THRESHOLD and p < alpha:
                    flagged.append({"feature": names[j],
                                    "ks_stat": round(float(d), 3),
                                    "p_value": float(f"{p:.2e}")})
            frac = len(flagged) / X.shape[1]
            if frac >= self._FRAC_WARN:
                anomalies[str(cls)] = {
                    "inconsistent_features_pct": round(frac * 100, 1),
                    "top_features": flagged[:5],
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "d_threshold": self._D_THRESHOLD,
                                   "frac_warn": self._FRAC_WARN})


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Jensen-Shannon Divergence (between class pairs, per feature)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class JSDivergenceScanner(BaseScanner):
    """
    M5: JSD между распределениями признаков по классам (Lin, 1991).
    Слишком низкий JSD (классы неотличимы) → возможное загрязнение подменой меток.
    Работает только для датасетов с 2–10 классами.
    """
    name = "Stats: M5 Jensen-Shannon divergence"
    category = ScannerCategory.STATS

    _LOW_JSD_WARN = 0.05   # mean JSD below this → classes look suspiciously similar
    _N_BINS = 20

    def run(self, context: ScanContext) -> ScanResult:
        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})
        X = b.X_tabular()
        if X.shape[1] == 0:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No tabular features"})

        classes = b.classes()
        if len(classes) < 2 or len(classes) > 10:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Needs 2–10 classes, got {len(classes)}"})

        y = b.y
        names = b.tabular_feature_names()
        cls0, cls1 = classes[0], classes[1]
        X0 = _sample(X[y == cls0], _MAX_SAMPLE)
        X1 = _sample(X[y == cls1], _MAX_SAMPLE)

        jsds = []
        for j in range(X.shape[1]):
            lo = min(X0[:, j].min(), X1[:, j].min())
            hi = max(X0[:, j].max(), X1[:, j].max())
            if hi == lo:
                continue
            bins = np.linspace(lo, hi, self._N_BINS + 1)
            p = np.histogram(X0[:, j], bins=bins, density=True)[0] + 1e-9
            q = np.histogram(X1[:, j], bins=bins, density=True)[0] + 1e-9
            p /= p.sum(); q /= q.sum()
            jsds.append(float(jensenshannon(p, q)))

        if not jsds:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No computable features"})

        mean_jsd = float(np.mean(jsds))
        low_jsd_features = [names[j] for j, v in enumerate(jsds) if v < self._LOW_JSD_WARN]

        status = ScanStatus.HAND_CHECK if mean_jsd < self._LOW_JSD_WARN else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={
                              "compared_classes": [str(cls0), str(cls1)],
                              "mean_jsd": round(mean_jsd, 4),
                              "low_jsd_features": low_jsd_features[:10],
                              "warn_threshold": self._LOW_JSD_WARN,
                          })


# ─────────────────────────────────────────────────────────────────────────────
# M6 — Wasserstein distance (each class vs global distribution, per feature)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class WassersteinScanner(BaseScanner):
    """
    M6: Расстояние Вассерштейна-1 между каждым классом и глобальным распределением.
    Класс с аномально высоким расстоянием по многим признакам — вероятно, содержит
    инъецированные выбросы. Порог: среднее W > 0.5 по StandardScaled признакам.
    """
    name = "Stats: M6 Wasserstein distance"
    category = ScannerCategory.STATS

    _WARN_DISTANCE = 0.50

    def run(self, context: ScanContext) -> ScanResult:
        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})
        X = b.X_tabular()
        if X.shape[1] == 0:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No tabular features"})

        y = b.y
        classes = b.classes()
        if len(classes) > _MAX_CLASSES:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": f"Too many classes ({len(classes)})"})

        X_global = _sample(X, _MAX_SAMPLE * len(classes))
        anomalies: Dict[str, dict] = {}

        for cls in classes:
            X_c = _sample(X[y == cls], _MAX_SAMPLE)
            if len(X_c) < 10:
                continue
            dists = [
                float(wasserstein_distance(X_c[:, j], X_global[:, j]))
                for j in range(X.shape[1])
            ]
            mean_d = float(np.mean(dists))
            if mean_d > self._WARN_DISTANCE:
                top = sorted(
                    zip(b.tabular_feature_names(), dists), key=lambda t: -t[1]
                )[:5]
                anomalies[str(cls)] = {
                    "mean_wasserstein": round(mean_d, 3),
                    "top_features": [{"feature": n, "distance": round(d, 3)} for n, d in top],
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "warn_threshold": self._WARN_DISTANCE})


# ─────────────────────────────────────────────────────────────────────────────
# M7 — Chi-square test for categorical features
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class ChiSquareScanner(BaseScanner):
    """
    M7: Хи-квадрат для проверки связи категориальных признаков с меткой.
    Работает по сырым данным — без векторизации.
    Слабая связь (p >> 0.05) у признака, который должен быть предсказательным → загрязнение.
    """
    name = "Stats: M7 chi-square (categorical)"
    category = ScannerCategory.STATS

    _P_WEAK = 0.10   # p > this → feature lost its association with label (suspicious)

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.FAILED, passed=False,
                              details={"reason": "Dataset not loaded"})

        label_col = find_label_column(df)
        if label_col is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No label column found"})

        b = _bundle(context)
        cat_cols = b.meta.get("categorical_columns", []) if b else []
        if not cat_cols:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No categorical features"})

        results: Dict[str, dict] = {}
        weak: List[str] = []

        for col in cat_cols:
            try:
                ct = pd.crosstab(df[col], df[label_col])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p, dof, _ = chi2_contingency(ct)
                results[col] = {"chi2": round(float(chi2), 2),
                                "p_value": float(f"{p:.3e}"),
                                "dof": int(dof)}
                if p > self._P_WEAK:
                    weak.append(col)
            except Exception:
                continue

        status = ScanStatus.HAND_CHECK if weak else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"label_column": label_col,
                                   "categorical_features": results,
                                   "weak_association_features": weak,
                                   "p_weak_threshold": self._P_WEAK})
"""
Спектральные, алгебраические и глобальные статистические детекторы.

M14  Спектральные сигнатуры — бэкдор-направление через SVD по классам (Tran et al., NeurIPS 2018)
M15  Робастный PCA          — ошибка реконструкции как мера аномальности (Candès et al., 2011)
M16  Конформный тест        — пермутационный тест с теоретической гарантией (Sánchez et al., 2025)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.core.factory import register_scanner
from src.core.features import FeatureBundle, reduce_to_n_components
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MAX_CLASSES = 50
_MAX_SAMPLE = 2000   # for O(n²) methods


def _bundle(ctx: ScanContext) -> Optional[FeatureBundle]:
    return ctx.metadata.get("features")


def _sample_with_labels(
    X: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    idx = np.random.default_rng(42).choice(len(X), n, replace=False)
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# M14 — Spectral Signatures
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class SpectralSignaturesScanner(BaseScanner):
    """
    M14: Спектральные сигнатуры (Tran, Li, Madry, NeurIPS 2018).
    Центрирует матрицу признаков класса, вычисляет ведущий правый сингулярный вектор v1.
    Бэкдор-примеры имеют значительно большую проекцию на v1, чем чистые.
    HAND_CHECK если отношение P95/P50 проекционных оценок превышает порог.

    Порог 20 стоит выше естественного P95/P50 для χ²₁ ≈ 8–9, чтобы не срабатывать на чистых данных.
    """
    name = "Stats: M14 spectral signatures"
    category = ScannerCategory.STATS

    _EPSILON = 0.10    # верхняя доля для пометки — Tran et al. используют ε=0.05–0.15
    _GAP_WARN = 20.0   # > 20: выше естественного P95/P50 для центрированных гауссовых данных (~8–9)

    def run(self, context: ScanContext) -> ScanResult:
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
            X_c = b.X[y == cls]
            if len(X_c) < 10:
                continue
            X_c = reduce_to_n_components(X_c, min(50, X_c.shape[0] - 1)) \
                if X_c.shape[1] > 50 else X_c
            X_centred = X_c - X_c.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            v1 = Vt[0]
            scores = (X_centred @ v1) ** 2
            p50 = float(np.percentile(scores, 50))
            p95 = float(np.percentile(scores, 95))
            gap = p95 / (p50 + 1e-10)
            if gap > self._GAP_WARN:
                anomalies[str(cls)] = {
                    "spectral_gap": round(float(gap), 2),
                    "p50_score": round(float(p50), 4),
                    "p95_score": round(float(p95), 4),
                    "top_epsilon_fraction": self._EPSILON,
                }

        status = ScanStatus.HAND_CHECK if anomalies else ScanStatus.PASSED
        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={"anomalous_classes": anomalies,
                                   "gap_warn_threshold": self._GAP_WARN})


# ─────────────────────────────────────────────────────────────────────────────
# M15 — Robust PCA (simplified via PCA reconstruction error)
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class RobustPCAScanner(BaseScanner):
    """
    M15: Ошибка реконструкции PCA как приближение к разреженной компоненте S
    из разложения X = L + S (Candès et al., JACM 2011).
    Образцы с высокой ошибкой — кандидаты в аномалии.
    Порог: медиана + 3.5×MAD (критерий Iglewicz & Hoaglin, тот же что в M1).
    """
    name = "Stats: M15 robust PCA"
    category = ScannerCategory.STATS

    _EXPLAINED_VAR = 0.90   # keep enough components to explain 90 % of variance
    _MAX_COMPONENTS = 50
    _OUTLIER_RATE_WARN = 0.08

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.decomposition import PCA

        b = _bundle(context)
        if b is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle not available"})

        X = b.X
        n, p = X.shape
        if n < 20 or p < 2:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Too few samples or features"})

        n_components = min(self._MAX_COMPONENTS, p, n - 1)
        pca = PCA(n_components=n_components, random_state=42)
        X_r = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_r)
        errors = np.linalg.norm(X - X_rec, axis=1)

        # критерий Iglewicz & Hoaglin: отклонение > 3.5 MAD от медианы → выброс
        med = np.median(errors)
        mad = np.median(np.abs(errors - med))
        threshold = float(med + 3.5 * mad)

        outlier_rate = float((errors > threshold).mean())
        top_indices: List[int] = np.argsort(errors)[-10:][::-1].tolist()

        expl = float(np.sum(pca.explained_variance_ratio_))
        status = ScanStatus.HAND_CHECK if outlier_rate > self._OUTLIER_RATE_WARN else ScanStatus.PASSED

        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=True,
                          details={
                              "n_components": n_components,
                              "explained_variance": round(expl, 3),
                              "reconstruction_threshold": round(threshold, 4),
                              "outlier_rate_pct": round(outlier_rate * 100, 2),
                              "warn_rate_pct": self._OUTLIER_RATE_WARN * 100,
                              "top_anomalous_indices": top_indices,
                          })


# ─────────────────────────────────────────────────────────────────────────────
# M16 — Conformal Separability Test
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class ConformalSeparabilityScanner(BaseScanner):
    """
    M16: Конформный тест сепарабельности (Sánchez et al., arXiv:2501.11795).

    Теорема: если атака отравления эффективна против любого классификатора,
    датасет обязательно проявит детектируемое нарушение сепарабельности.

    Мера несоответствия для образца i:
        score(i) = d(i, ближайший сосед того же класса) /
                   d(i, ближайший сосед другого класса)

    score > 1 → точка ближе к чужому классу, чем к своему.
    Глобальная статистика = доля таких точек.
    Значимость оценивается пермутационным тестом (300 пермутаций).
    FAILED если p-value < 0.10.
    """
    name = "Stats: M16 conformal separability"
    category = ScannerCategory.STATS

    _N_PERMS = 300
    _ALPHA = 0.10   # 0.10 вместо 0.05: более чувствителен к умеренным нарушениям сепарабельности

    def run(self, context: ScanContext) -> ScanResult:
        from sklearn.neighbors import NearestNeighbors

        b = _bundle(context)
        if b is None or b.y is None:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Bundle or labels unavailable"})

        X_full, y_full = b.X, b.y
        if len(np.unique(y_full)) < 2:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "Need at least 2 classes"})

        X, y = _sample_with_labels(X_full, y_full, _MAX_SAMPLE)
        X = reduce_to_n_components(X, 30) if X.shape[1] > 30 else X

        def _statistic(y_perm: np.ndarray) -> float:
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
            dists_all, idx_all = nbrs.kneighbors(X)

            scores = []
            for i in range(len(X)):
                # nearest same-class neighbour (may not be index 1 if class is very small)
                same = [j for j in idx_all[i, 1:] if y_perm[j] == y_perm[i]]
                diff = [j for j in idx_all[i, 1:] if y_perm[j] != y_perm[i]]
                if not same or not diff:
                    continue
                d_same = float(np.linalg.norm(X[i] - X[same[0]]))
                d_diff = float(np.linalg.norm(X[i] - X[diff[0]]))
                scores.append(d_same / (d_diff + 1e-10))
            return float(np.mean(np.array(scores) > 1.0)) if scores else 0.0

        # batch KNN с k > 1 — за один проход получаем и соседей своего, и чужого класса
        k = min(20, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
        dists_all, idx_all = nbrs.kneighbors(X)

        def _fast_statistic(y_perm: np.ndarray) -> float:
            scores = []
            for i in range(len(X)):
                neighbours = idx_all[i, 1:]
                same = neighbours[y_perm[neighbours] == y_perm[i]]
                diff = neighbours[y_perm[neighbours] != y_perm[i]]
                if len(same) == 0 or len(diff) == 0:
                    continue
                d_same = dists_all[i, np.where(idx_all[i, 1:] == same[0])[0][0] + 1]
                d_diff = dists_all[i, np.where(idx_all[i, 1:] == diff[0])[0][0] + 1]
                scores.append(float(d_same) / (float(d_diff) + 1e-10))
            return float(np.mean(np.array(scores) > 1.0)) if scores else 0.0

        observed = _fast_statistic(y)

        rng = np.random.default_rng(42)
        perm_stats = [_fast_statistic(rng.permutation(y))
                      for _ in range(self._N_PERMS)]

        p_value = float(np.mean(np.array(perm_stats) >= observed))

        if p_value < self._ALPHA:
            status, passed = ScanStatus.FAILED, False
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=passed,
                          details={
                              "observed_statistic": round(observed, 4),
                              "p_value": round(p_value, 4),
                              "alpha": self._ALPHA,
                              "n_permutations": self._N_PERMS,
                              "n_samples_used": len(X),
                              "interpretation": (
                                  "Классы плохо разделены — возможно отравление (p < alpha)"
                                  if not passed else
                                  "Разделимость классов соответствует чистым данным"
                              ),
                          })
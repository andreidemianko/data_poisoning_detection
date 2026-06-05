from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.core.factory import register_scanner
from src.core.features import find_label_column as _find_label_column
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory


@register_scanner
class MissingValuesScanner(BaseScanner):
    name = "Stats: missing values"
    category = ScannerCategory.STATS

    _WARN_THRESHOLD = 0.05
    _FAIL_THRESHOLD = 0.50

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        null_rates = df.isnull().mean()
        fail_cols: Dict[str, float] = {
            c: round(float(r) * 100, 2)
            for c, r in null_rates.items()
            if r >= self._FAIL_THRESHOLD
        }
        warn_cols: Dict[str, float] = {
            c: round(float(r) * 100, 2)
            for c, r in null_rates.items()
            if self._WARN_THRESHOLD <= r < self._FAIL_THRESHOLD
        }

        if fail_cols:
            status, passed = ScanStatus.FAILED, False
        elif warn_cols:
            status, passed = ScanStatus.HAND_CHECK, True  # предупреждение, пайплайн не блокируется
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "total_rows": len(df),
                "total_nulls": int(df.isnull().sum().sum()),
                "failed_columns_null_pct": fail_cols,
                "warn_columns_null_pct": warn_cols,
            },
        )


@register_scanner
class DuplicateRowsScanner(BaseScanner):
    name = "Stats: duplicate rows"
    category = ScannerCategory.STATS

    _WARN_THRESHOLD = 0.01
    _FAIL_THRESHOLD = 0.35  # повышен: чистые NLP-датасеты бывают с 20-27 % естественных дублей

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        n = len(df)
        dup_count = int(df.duplicated().sum())
        dup_rate = dup_count / max(n, 1)

        if dup_rate >= self._FAIL_THRESHOLD:
            status, passed = ScanStatus.FAILED, False
        elif dup_rate >= self._WARN_THRESHOLD:
            status, passed = ScanStatus.HAND_CHECK, True
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "total_rows": n,
                "duplicate_rows": dup_count,
                "duplicate_rate_pct": round(dup_rate * 100, 3),
            },
        )


@register_scanner
class ExactLabelConflictScanner(BaseScanner):
    """Строки с одинаковыми признаками, но разными метками — прямой индикатор подмены меток."""
    name = "Stats: exact label conflict"
    category = ScannerCategory.STATS

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        label_col = _find_label_column(df)
        if label_col is None:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No label column found"},
            )

        feat_cols = [c for c in df.columns if c != label_col]

        # pandas внутренний хэш строк — быстро и без коллизий на практике
        row_hashes = pd.util.hash_pandas_object(df[feat_cols], index=False)
        hash_label = pd.DataFrame({"_hash": row_hashes, "_label": df[label_col].values})

        grouped = hash_label.groupby("_hash")["_label"].nunique()
        conflict_hashes = grouped[grouped > 1].index

        n_conflict_groups = int(len(conflict_hashes))
        n_conflict_rows = int(row_hashes.isin(conflict_hashes).sum())

        # до 2% конфликтов — граничные случаи бывают в реальных датасетах (разметка с несогласием)
        conflict_rate = n_conflict_rows / max(len(df), 1)
        if conflict_rate >= 0.02:
            status, passed = ScanStatus.FAILED, False
        elif n_conflict_groups > 0:
            status, passed = ScanStatus.HAND_CHECK, True
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "label_column": label_col,
                "conflict_groups": n_conflict_groups,
                "conflicting_rows": n_conflict_rows,
                "conflict_rate_pct": round(conflict_rate * 100, 2),
            },
        )


@register_scanner
class ConstantFeaturesScanner(BaseScanner):
    """Колонки с единственным уникальным значением — признак возможного повреждения данных."""
    name = "Stats: constant features"
    category = ScannerCategory.STATS

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        label_col = _find_label_column(df)
        feat_cols = [c for c in df.columns if c != label_col]
        constant = [c for c in feat_cols if df[c].nunique(dropna=True) <= 1]

        status = ScanStatus.PASSED if not constant else ScanStatus.HAND_CHECK

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=True,
            details={
                "constant_columns": constant,
                "count": len(constant),
            },
        )


@register_scanner
class ClassBalanceScanner(BaseScanner):
    """
    Проверяет дисбаланс классов. Всегда HAND_CHECK — сильный дисбаланс может быть
    естественным (фрод-детекция, медицина), оценить его без референсного датасета нельзя.
    """
    name = "Stats: class balance"
    category = ScannerCategory.STATS

    _WARN_RATIO = 5.0

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        label_col = _find_label_column(df)
        if label_col is None:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No label column found"},
            )

        counts = df[label_col].value_counts()
        min_count = int(counts.min()) if not counts.empty else 0
        max_count = int(counts.max()) if not counts.empty else 0
        imbalance_ratio = (max_count / max(min_count, 1)) if counts.size > 0 else 0.0

        status = ScanStatus.HAND_CHECK if imbalance_ratio > self._WARN_RATIO else ScanStatus.PASSED

        return ScanResult(
            name=self.name, category=self.category,
            status=status, passed=True,
            details={
                "label_column": label_col,
                "class_counts": counts.to_dict(),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "warn_ratio": self._WARN_RATIO,
            },
        )

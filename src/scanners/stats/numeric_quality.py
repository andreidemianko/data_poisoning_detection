from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.core.factory import register_scanner
from src.core.features import find_label_column as _find_label_column
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MIN_UNIQUE_FOR_CONTINUOUS = 10
_MIN_FREQ_RATE = 0.01  # значение должно встречаться хотя бы в 1% строк
_IQR_DISTANCE_THRESHOLD = 3.0   # значение должно быть СТРОГО дальше 3 IQR от Q1/Q3
# Граница Tukey для "extreme outlier" — ровно 3×IQR.
# Синтетические сентинелы (Age=99, CreditScore=85000) существенно превышают её
# и одновременно встречаются с аномально высокой частотой.


@register_scanner
class NumericSentinelValueScanner(BaseScanner):
    """
    Ищет искусственно вброшенные сентинел-значения: появляются часто (≥1% строк)
    и при этом далеко за пределами нормального распределения (> 3 IQR от Q1/Q3).
    Ловит атаки типа «подменить CreditScore на 85000» или «вписать Age=99».
    """
    name = "Stats: numeric sentinel values"
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
        num_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != label_col and df[c].nunique(dropna=True) >= _MIN_UNIQUE_FOR_CONTINUOUS
        ]

        if not num_cols:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No continuous numeric columns found"},
            )

        n = len(df)
        min_count = max(1, int(_MIN_FREQ_RATE * n))
        flagged: Dict[str, dict] = {}

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue

            value_counts = series.value_counts()
            suspicious_vals: List[dict] = []

            for val, cnt in value_counts.items():
                if cnt < min_count:
                    break  # sorted descending, no point continuing
                distance = max(
                    (float(val) - q3) / iqr,
                    (q1 - float(val)) / iqr,
                    0.0,
                )
                if distance > _IQR_DISTANCE_THRESHOLD:   # строго больше: естественные выбросы на границе Tukey дают ровно 3.0
                    suspicious_vals.append({
                        "value": float(val),
                        "count": int(cnt),
                        "freq_pct": round(cnt / n * 100, 2),
                        "iqr_distance": round(distance, 1),
                    })

            if suspicious_vals:
                flagged[col] = {
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "sentinel_values": suspicious_vals,
                }

        passed = len(flagged) == 0
        status = ScanStatus.PASSED if passed else ScanStatus.FAILED

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "columns_checked": len(num_cols),
                "flagged_columns": flagged,
            },
        )

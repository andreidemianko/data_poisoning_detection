from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from src.core.factory import register_scanner
from src.core.features import find_label_column as _find_label_column
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

_MIN_TEXT_AVG_LEN = 25


def _find_text_columns(df: pd.DataFrame) -> List[str]:
    text_cols = []
    for col in df.select_dtypes(include="object").columns:
        avg_len = df[col].dropna().str.len().mean()
        if avg_len is not None and avg_len >= _MIN_TEXT_AVG_LEN:
            text_cols.append(col)
    return text_cols


@register_scanner
class EmptyTextsScanner(BaseScanner):
    """Пустые или состоящие только из пробелов строки в текстовых колонках."""
    name = "Stats: empty texts"
    category = ScannerCategory.STATS

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        text_cols = _find_text_columns(df)
        if not text_cols:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No text columns found"},
            )

        empty_per_col: Dict[str, int] = {}
        for col in text_cols:
            n_empty = int(df[col].fillna("").str.strip().eq("").sum())
            if n_empty > 0:
                empty_per_col[col] = n_empty

        passed = len(empty_per_col) == 0
        status = ScanStatus.PASSED if passed else ScanStatus.FAILED

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "text_columns_checked": text_cols,
                "empty_per_column": empty_per_col,
            },
        )


@register_scanner
class TextLengthAnomalyScanner(BaseScanner):
    """Аномально короткие или длинные тексты по методу IQR — срезанный или дополненный контент."""
    name = "Stats: text length anomalies"
    category = ScannerCategory.STATS

    _IQR_MULTIPLIER = 5.0

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        text_cols = _find_text_columns(df)
        if not text_cols:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No text columns found"},
            )

        col_stats: Dict[str, dict] = {}
        anomalous_cols: Dict[str, int] = {}

        for col in text_cols:
            lengths = df[col].dropna().str.len()
            q1 = float(lengths.quantile(0.25))
            q3 = float(lengths.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - self._IQR_MULTIPLIER * iqr
            upper = q3 + self._IQR_MULTIPLIER * iqr
            n_anomalies = int(((lengths < lower) | (lengths > upper)).sum())

            col_stats[col] = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": round(float(lengths.mean()), 1),
                "lower_fence": round(lower, 1),
                "upper_fence": round(upper, 1),
                "anomaly_count": n_anomalies,
                "anomaly_rate_pct": round(n_anomalies / max(len(lengths), 1) * 100, 2),
            }
            if n_anomalies > 0:
                anomalous_cols[col] = n_anomalies

        status = ScanStatus.PASSED if not anomalous_cols else ScanStatus.HAND_CHECK

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=True,
            details={
                "column_stats": col_stats,
                "columns_with_anomalies": anomalous_cols,
            },
        )


@register_scanner
class TextLabelConflictScanner(BaseScanner):
    """Одинаковые тексты с разными метками — прямой индикатор подмены меток."""
    name = "Stats: text-label conflict"
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

        text_cols = _find_text_columns(df)
        if not text_cols:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No text columns found"},
            )

        conflicts: Dict[str, dict] = {}
        any_hard_fail = False
        for col in text_cols:
            pair = df[[col, label_col]].dropna()
            label_counts = pair.groupby(col)[label_col].nunique()
            n_unique = int(label_counts.shape[0])
            n_conflicted = int((label_counts > 1).sum())
            if n_conflicted > 0:
                conflict_rate = n_conflicted / max(n_unique, 1)
                # < 1% конфликтов — расхождение разметчиков, норма; ≥ 1% — подозрительно
                is_hard_fail = conflict_rate >= 0.01
                if is_hard_fail:
                    any_hard_fail = True
                conflicts[col] = {
                    "conflicting_texts": n_conflicted,
                    "total_unique_texts": n_unique,
                    "conflict_rate_pct": round(conflict_rate * 100, 2),
                    "severity": "high" if is_hard_fail else "low",
                }

        if any_hard_fail:
            status, passed = ScanStatus.FAILED, False
        elif conflicts:
            status, passed = ScanStatus.HAND_CHECK, True
        else:
            status, passed = ScanStatus.PASSED, True

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "label_column": label_col,
                "text_columns_checked": text_cols,
                "conflicts": conflicts,
            },
        )


# Токены с перемежающимися буквами и цифрами — характерная сигнатура синтетических бэкдор-триггеров (типа qx9b7zftrigger)
_MIXED_ALNUM = re.compile(r'(?:[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+)[a-zA-Z0-9]*')


@register_scanner
class SuspiciousTokenScanner(BaseScanner):
    """Бэкдор-триггеры: буквенно-цифровые токены, встречающиеся в >10% текстов."""
    name = "Stats: suspicious tokens"
    category = ScannerCategory.STATS

    _DOC_FREQ_THRESHOLD = 0.10

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.FAILED, passed=False,
                details={"reason": "Dataset not loaded"},
            )

        text_cols = _find_text_columns(df)
        if not text_cols:
            return ScanResult(
                name=self.name, category=self.category,
                status=ScanStatus.SKIPPED, passed=True,
                details={"reason": "No text columns found"},
            )

        # сканируем все текстовые колонки — триггер может быть инжектирован в любую из них
        # (например, в колонку instruction в датасетах instruction-tuning)
        n = len(df)
        all_suspicious: Dict[str, Dict[str, float]] = {}

        for col in text_cols:
            texts = df[col].dropna().astype(str)
            n_col = len(texts)
            doc_freq: Counter = Counter()
            for text in texts:
                tokens = set(re.split(r"\s+", text.lower().strip()))
                tokens.discard("")
                doc_freq.update(tokens)

            threshold = self._DOC_FREQ_THRESHOLD * n_col
            col_suspicious = {
                tok: round(cnt / n_col * 100, 2)
                for tok, cnt in doc_freq.items()
                if cnt >= threshold and _MIXED_ALNUM.fullmatch(tok)
            }
            if col_suspicious:
                all_suspicious[col] = dict(sorted(col_suspicious.items(), key=lambda x: -x[1]))

        passed = len(all_suspicious) == 0
        status = ScanStatus.PASSED if passed else ScanStatus.FAILED

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "text_columns_scanned": text_cols,
                "total_samples": n,
                "doc_freq_threshold_pct": self._DOC_FREQ_THRESHOLD * 100,
                "suspicious_tokens_by_column": all_suspicious,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Non-ASCII character detector
# ─────────────────────────────────────────────────────────────────────────────
@register_scanner
class NonAsciiTextScanner(BaseScanner):
    """
    Обнаруживает не-ASCII символы в текстовых колонках датасетов, которые должны быть на английском.
    Ловит атаки с подменой букв кириллицей или омоглифами (e → е, a → а и т.д.).
    FAILED если доля не-ASCII символов в любой текстовой колонке превышает 0.5%.
    """
    name = "Stats: non-ASCII characters"
    category = ScannerCategory.STATS

    _FAIL_RATE = 0.005   # 0.5 % non-ASCII chars → suspicious

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.FAILED, passed=False,
                              details={"reason": "Dataset not loaded"})

        text_cols = _find_text_columns(df)
        if not text_cols:
            return ScanResult(name=self.name, category=self.category,
                              status=ScanStatus.SKIPPED, passed=True,
                              details={"reason": "No text columns found"})

        flagged: Dict[str, dict] = {}
        for col in text_cols:
            texts = df[col].fillna("").astype(str)
            total_chars = int(texts.str.len().sum())
            if total_chars == 0:
                continue
            non_ascii = int(texts.apply(
                lambda s: sum(1 for c in s if ord(c) > 127)
            ).sum())
            rate = non_ascii / total_chars
            if rate >= self._FAIL_RATE:
                flagged[col] = {
                    "non_ascii_chars": non_ascii,
                    "total_chars": total_chars,
                    "non_ascii_rate_pct": round(rate * 100, 3),
                }

        passed = len(flagged) == 0
        status = ScanStatus.PASSED if passed else ScanStatus.FAILED

        return ScanResult(name=self.name, category=self.category,
                          status=status, passed=passed,
                          details={
                              "text_columns_checked": text_cols,
                              "flagged_columns": flagged,
                              "fail_rate_pct": self._FAIL_RATE * 100,
                          })

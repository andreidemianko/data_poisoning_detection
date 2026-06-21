from __future__ import annotations
import json

import numpy as np
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import PoisoningConfig
from ..models import Category, DetectorInfo, Finding, Severity
from ..utils import safe_ratio
from .base import Profiler


LABEL_COLUMN_PATTERN = re.compile(
    r"(?:^|[_ .-])("
    r"label|target|class|y|category|outcome|is_fraud|fraud|malicious|benign"
    r")(?:$|[_ .-])",
    re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_@$#./\\:-]{3,}")


@dataclass(frozen=True)
class LabelColumnInfo:
    """
    Description of a likely label column.
    """

    name: str
    unique_count: int
    non_null_count: int


class PoisoningProfiler(Profiler):
    """
    Heuristic profiler for dataset poisoning and data-quality risks.

    These checks are not proof of poisoning. They are designed to flag suspicious
    dataset properties that deserve review.
    """

    name = "poisoning_profiler"

    def __init__(self, config: PoisoningConfig | None = None) -> None:
        self.config = config or PoisoningConfig()

    def describe(self) -> DetectorInfo:
        """
        Return report-friendly profiler metadata.
        """

        return DetectorInfo(
            name=self.name,
            status="available",
            metadata={
                "min_rows_for_statistical_checks": self.config.min_rows_for_statistical_checks,
                "class_imbalance_warn_ratio": self.config.class_imbalance_warn_ratio,
                "class_imbalance_block_ratio": self.config.class_imbalance_block_ratio,
                "duplicate_rows_warn_ratio": self.config.duplicate_rows_warn_ratio,
                "duplicate_rows_block_ratio": self.config.duplicate_rows_block_ratio,
                "numeric_outlier_zscore": self.config.numeric_outlier_zscore,
                "rare_token_min_count": self.config.rare_token_min_count,
                "rare_token_label_correlation": self.config.rare_token_label_correlation,
            },
        )

    def scan_frame(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Run all poisoning and quality heuristics for a dataframe.
        """

        frame = self._sanitize_frame_for_hashing(frame)

        findings: list[Finding] = []

        if frame.empty:
            findings.append(
                Finding(
                    category=Category.QUALITY,
                    subtype="EMPTY_DATASET",
                    rule_id="quality.empty_dataset",
                    severity=Severity.REVIEW,
                    message="Dataset is empty.",
                    detector=self.name,
                    confidence=1.0,
                )
            )
            return findings

        findings.extend(self._check_small_dataset(frame))
        findings.extend(self._check_duplicate_rows(frame))
        findings.extend(self._check_constant_features(frame))
        findings.extend(self._check_numeric_outliers(frame))

        label_columns = self._find_label_columns(frame)

        for label_column in label_columns:
            findings.extend(self._check_class_imbalance(frame, label_column))
            findings.extend(self._check_rare_token_label_correlation(frame, label_column))

        return findings

    def _check_small_dataset(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Flag datasets that are too small for reliable statistical checks.
        """

        row_count = len(frame)

        if row_count >= self.config.min_rows_for_statistical_checks:
            return []

        return [
            Finding(
                category=Category.QUALITY,
                subtype="SMALL_DATASET",
                rule_id="quality.small_dataset",
                severity=Severity.WARN,
                message=(
                    f"Dataset has only {row_count} rows. "
                    "Statistical poisoning checks may be unreliable."
                ),
                detector=self.name,
                confidence=0.8,
                metadata={
                    "rows": row_count,
                    "min_rows_for_statistical_checks": self.config.min_rows_for_statistical_checks,
                },
            )
        ]

    def _check_duplicate_rows(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Flag high duplicate-row ratios.

        Duplicates can be benign, but high duplicate ratios can also indicate
        oversampling, poisoning, or accidental dataset construction issues.
        """

        row_count = len(frame)

        if row_count <= 1:
            return []

        duplicate_count = int(frame.duplicated().sum())
        duplicate_ratio = safe_ratio(duplicate_count, row_count)

        if duplicate_ratio < self.config.duplicate_rows_warn_ratio:
            return []

        severity = (
            Severity.BLOCK
            if duplicate_ratio >= self.config.duplicate_rows_block_ratio
            else Severity.REVIEW
        )

        category = (
            Category.DATA_POISONING
            if duplicate_ratio >= self.config.duplicate_rows_block_ratio
            else Category.QUALITY
        )

        return [
            Finding(
                category=category,
                subtype="DUPLICATE_ROWS",
                rule_id="poisoning.duplicate_rows",
                severity=severity,
                message=f"High duplicate row ratio detected: {duplicate_ratio:.2%}.",
                detector=self.name,
                confidence=min(1.0, duplicate_ratio + 0.25),
                metadata={
                    "row_count": row_count,
                    "duplicate_count": duplicate_count,
                    "duplicate_ratio": duplicate_ratio,
                    "warn_ratio": self.config.duplicate_rows_warn_ratio,
                    "block_ratio": self.config.duplicate_rows_block_ratio,
                },
            )
        ]

    def _check_constant_features(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Flag columns with a single unique non-null value.

        Constant features may be harmless metadata, but they can also indicate
        broken preprocessing or hidden trigger columns.
        """

        findings: list[Finding] = []

        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()

            if non_null.empty:
                continue

            unique_count = int(non_null.nunique(dropna=True))

            if unique_count > self.config.constant_feature_warn_unique_values:
                continue

            findings.append(
                Finding(
                    category=Category.QUALITY,
                    subtype="CONSTANT_FEATURE",
                    rule_id="quality.constant_feature",
                    severity=Severity.WARN,
                    message=f"Column '{column}' has a constant non-null value.",
                    detector=self.name,
                    confidence=0.6,
                    column=str(column),
                    metadata={
                        "unique_count": unique_count,
                        "non_null_count": int(non_null.shape[0]),
                    },
                )
            )

        return findings

    def _check_numeric_outliers(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Flag numeric columns with extreme z-score outliers.
        """

        findings: list[Finding] = []

        numeric_frame = frame.select_dtypes(include="number")

        for column in numeric_frame.columns:
            series = numeric_frame[column].dropna()

            if series.shape[0] < self.config.min_rows_for_statistical_checks:
                continue

            mean = float(series.mean())
            std = float(series.std(ddof=0))

            if std == 0 or math.isnan(std):
                continue

            zscores = ((series - mean).abs() / std)
            max_zscore = float(zscores.max())

            if max_zscore < self.config.numeric_outlier_zscore:
                continue

            row_index = zscores.idxmax()

            findings.append(
                Finding(
                    category=Category.QUALITY,
                    subtype="NUMERIC_EXTREME_OUTLIER",
                    rule_id="poisoning.numeric_extreme_outlier",
                    severity=Severity.WARN,
                    message=(
                        f"Column '{column}' contains an extreme numeric outlier "
                        f"with z-score {max_zscore:.2f}."
                    ),
                    detector=self.name,
                    confidence=min(1.0, max_zscore / (self.config.numeric_outlier_zscore * 2)),
                    column=str(column),
                    row_index=int(row_index) if isinstance(row_index, int) else str(row_index),
                    metadata={
                        "mean": mean,
                        "std": std,
                        "max_zscore": max_zscore,
                        "threshold": self.config.numeric_outlier_zscore,
                    },
                )
            )

        return findings

    def _check_class_imbalance(self, frame: pd.DataFrame, label_column: LabelColumnInfo) -> list[Finding]:
        """
        Flag highly imbalanced label distributions.

        Moderate imbalance is reported as data quality. Extreme imbalance is
        reported as potential data poisoning.
        """

        series = frame[label_column.name].dropna()

        if series.empty:
            return []

        counts = series.value_counts(dropna=True)
        top_count = int(counts.iloc[0])
        total = int(counts.sum())
        top_ratio = safe_ratio(top_count, total)

        if top_ratio < self.config.class_imbalance_warn_ratio:
            return []

        category = (
            Category.DATA_POISONING
            if top_ratio >= self.config.class_imbalance_block_ratio
            else Category.QUALITY
        )

        severity = (
            Severity.REVIEW
            if top_ratio >= self.config.class_imbalance_block_ratio
            else Severity.WARN
        )

        return [
            Finding(
                category=category,
                subtype="CLASS_IMBALANCE",
                rule_id="poisoning.class_imbalance",
                severity=severity,
                message=(
                    f"Label column '{label_column.name}' is highly imbalanced. "
                    f"Top class ratio: {top_ratio:.2%}."
                ),
                detector=self.name,
                confidence=min(1.0, top_ratio),
                column=label_column.name,
                metadata={
                    "label_column": label_column.name,
                    "top_count": top_count,
                    "total": total,
                    "top_ratio": top_ratio,
                    "class_counts": {str(key): int(value) for key, value in counts.items()},
                    "warn_ratio": self.config.class_imbalance_warn_ratio,
                    "block_ratio": self.config.class_imbalance_block_ratio,
                },
            )
        ]

    def _check_rare_token_label_correlation(
        self,
        frame: pd.DataFrame,
        label_column: LabelColumnInfo,
    ) -> list[Finding]:
        """
        Detect rare tokens that are strongly correlated with one label.

        This can catch simple trigger-token poisoning patterns, for example a
        rare marker that appears almost exclusively in malicious/target class
        rows.
        """

        text_columns = [
            column
            for column in frame.columns
            if column != label_column.name and self._is_text_series(frame[column])
        ]

        if not text_columns:
            return []

        token_to_labels: dict[str, Counter[Any]] = defaultdict(Counter)

        for _, row in frame.iterrows():
            label = self._hashable_value(row.get(label_column.name))

            if label is None:
                continue

            row_tokens: set[str] = set()

            for column in text_columns:
                value = row.get(column)

                if not isinstance(value, str):
                    continue

                row_tokens.update(self._tokens(value))

            for token in row_tokens:
                token_to_labels[token][label] += 1

        findings: list[Finding] = []

        for token, label_counts in token_to_labels.items():
            total_count = sum(label_counts.values())

            if total_count < self.config.rare_token_min_count:
                continue

            top_label, top_count = label_counts.most_common(1)[0]
            top_ratio = safe_ratio(top_count, total_count)

            if top_ratio < self.config.rare_token_label_correlation:
                continue

            findings.append(
                Finding(
                    category=Category.QUALITY,
                    subtype="RARE_TOKEN_LABEL_CORRELATION",
                    rule_id="poisoning.rare_token_label_correlation",
                    severity=Severity.REVIEW,
                    message=f"Rare token is strongly correlated with label '{top_label}'.",
                    detector=self.name,
                    confidence=min(1.0, top_ratio),
                    column=label_column.name,
                    metadata={
                        "label_column": label_column.name,
                        "token_sha256": self._token_hash(token),
                        "token_count": total_count,
                        "top_label": str(top_label),
                        "top_label_count": int(top_count),
                        "top_ratio": top_ratio,
                        "threshold": self.config.rare_token_label_correlation,
                    },
                )
            )

        return findings

    def _find_label_columns(self, frame: pd.DataFrame) -> list[LabelColumnInfo]:
        """
        Find likely label columns by name and cardinality.
        """

        label_columns: list[LabelColumnInfo] = []

        for column in frame.columns:
            column_name = str(column)

            series = frame[column].dropna()
            if series.empty:
                continue

            series = series.map(self._hashable_value)

            unique_count = int(series.nunique(dropna=True))
            non_null_count = int(series.shape[0])

            if LABEL_COLUMN_PATTERN.search(column_name):
                label_columns.append(
                    LabelColumnInfo(
                        name=column_name,
                        unique_count=unique_count,
                        non_null_count=non_null_count,
                    )
                )
                continue

            # Heuristic fallback for common classification labels:
            # low-cardinality columns in reasonably sized datasets.
            if 2 <= unique_count <= 20 and non_null_count >= self.config.min_rows_for_statistical_checks:
                if unique_count <= max(2, int(math.sqrt(non_null_count))):
                    label_columns.append(
                        LabelColumnInfo(
                            name=column_name,
                            unique_count=unique_count,
                            non_null_count=non_null_count,
                        )
                    )

        return label_columns

    @staticmethod
    def _is_text_series(series: pd.Series) -> bool:
        """
        Return True for columns that may contain text values.
        """

        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """
        Extract normalized tokens from text.
        """

        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    @staticmethod
    def _token_hash(token: str) -> str:
        """
        Hash a token before including it in reports.

        This avoids leaking potentially sensitive trigger tokens directly.
        """

        from ..utils import value_fingerprint

        return value_fingerprint(token)

    @staticmethod
    def _hashable_value(value: Any) -> Any:
        """
        Convert nested/list/ndarray values into stable hashable representation
        for Counter/set/value_counts/grouping logic.
        """

        if value is None:
            return None

        if isinstance(value, float) and math.isnan(value):
            return None

        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist(), ensure_ascii=False, sort_keys=True, default=str)

        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False, sort_keys=True, default=str)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

        return value

    @classmethod
    def _sanitize_frame_for_hashing(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert unhashable object values to stable strings before profiler checks.
        """

        safe = frame.copy()

        for column in safe.columns:
            if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
                safe[column] = safe[column].map(cls._hashable_value)

        return safe
"""
Пресканер, запускается первым в категории stats (имя файла 'f' < 'n/s/t').
Векторизует датасет и сохраняет FeatureBundle в context.metadata["features"].
При ошибке возвращает SKIPPED, а не FAILED — остальные сканеры продолжают работу.
"""
from __future__ import annotations

import pandas as pd

from src.core.factory import register_scanner
from src.core.features import find_label_column, vectorize
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory


@register_scanner
class FeatureVectorizerScanner(BaseScanner):
    """Векторизует датасет и кладёт FeatureBundle в context.metadata['features']."""

    name = "Stats: feature vectorizer"
    category = ScannerCategory.STATS

    def run(self, context: ScanContext) -> ScanResult:
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame):
            return ScanResult(
                name=self.name,
                category=self.category,
                status=ScanStatus.SKIPPED,
                passed=True,
                details={"reason": "Dataset not loaded"},
            )

        label_col = find_label_column(df)
        try:
            bundle = vectorize(df, label_col)
        except Exception as exc:
            return ScanResult(
                name=self.name,
                category=self.category,
                status=ScanStatus.SKIPPED,
                passed=True,
                details={"reason": f"Vectorization failed: {exc}"},
            )

        context.metadata["features"] = bundle

        classes = bundle.classes()
        return ScanResult(
            name=self.name,
            category=self.category,
            status=ScanStatus.PASSED,
            passed=True,
            details={
                "dataset_type": bundle.dataset_type,
                "n_samples": bundle.n_samples,
                "n_features": bundle.n_features,
                "n_numeric_features": bundle.meta.get("n_numeric", 0),
                "n_categorical_features": bundle.meta.get("n_categorical", 0),
                "n_tfidf_features": bundle.meta.get("n_text_tfidf", 0),
                "text_column": bundle.meta.get("text_column"),
                "label_column": bundle.label_col,
                "n_classes": int(len(classes)) if classes is not None else None,
            },
        )
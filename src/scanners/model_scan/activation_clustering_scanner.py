from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class ActivationClusteringScanner(BaseScanner):
    """Adapter: Activation Clustering (model-level) with silhouette/size gate."""
    name = "Model: Activation Clustering"
    category = ScannerCategory.MODEL
    sil_review = 0.45  # тесный обособленный кластер (silhouette) -> REVIEW; эвристика, тюнится

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.model_level import prepare_model, mlp_forward, activation_clustering
            from src.detectors.data_level import top_indices
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        try:
            layers, X, y, label_col = prepare_model(context.model_state, df)
        except ValueError as exc:
            return H.skip(self, str(exc))
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "model prepare failed", exc)
        if len(X) < 20:
            return H.skip(self, f"too few rows ({len(X)})")
        repr_, _ = mlp_forward(layers, X)
        flags, scores, meta = activation_clustering(repr_, y)
        sil = meta.get("max_minority_silhouette", -1.0)
        det = {"label_column": label_col, "n_rows": len(X), "flagged": int(flags.sum()),
               "top_suspicious_rows": top_indices(scores),
               "note": "model-level, approximate/uncalibrated -> REVIEW not auto-block", **meta}
        if sil >= self.sil_review:  # тесный обособленный кластер-меньшинство = след backdoor
            return H.review(self, verdict=f"tight minority cluster (silhouette={sil})", **det)
        return H.ok(self, verdict="no tight minority cluster", **det)

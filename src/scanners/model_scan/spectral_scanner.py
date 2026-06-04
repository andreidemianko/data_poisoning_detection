from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class SpectralScanner(BaseScanner):
    """Adapter: Spectral Signatures (model-level). Reconstructs MLP from state_dict."""
    name = "Model: Spectral Signatures"
    category = ScannerCategory.MODEL
    review_frac = 0.02

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.model_level import prepare_model, mlp_forward, spectral_scores
            from src.detectors.data_level import outlier_flags, top_indices
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
        s = spectral_scores(repr_, y)
        fl = outlier_flags(s)
        frac = float(fl.mean())
        det = {"label_column": label_col, "n_rows": len(X), "flagged": int(fl.sum()),
               "flagged_fraction": round(frac, 4), "top_suspicious_rows": top_indices(s),
               "note": "model-level, approximate/uncalibrated -> REVIEW not auto-block"}
        if frac >= self.review_frac:
            return H.review(self, verdict=f"{int(fl.sum())} spectral-outlier rows", **det)
        return H.ok(self, verdict="no spectral outliers", **det)

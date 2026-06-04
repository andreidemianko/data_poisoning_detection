from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class SecureLearnScanner(BaseScanner):
    """Adapter: SecureLearn per-class feature-outlier detector (data-level, tabular)."""
    name = "Stats: SecureLearn (per-class feature outliers)"
    category = ScannerCategory.STATS
    review_frac = 0.02  # доля выбросов для REVIEW (эвристика, тюнится)

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.data_level import extract_xy, securelearn_scores, outlier_flags, top_indices
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        try:
            X, y, label_col = extract_xy(df)
        except ValueError as exc:
            return H.skip(self, str(exc))
        if len(X) < 20:
            return H.skip(self, f"too few rows ({len(X)})")
        s = securelearn_scores(X, y)
        fl = outlier_flags(s)
        frac = float(fl.mean())
        det = {"label_column": label_col, "n_rows": len(X), "flagged": int(fl.sum()),
               "flagged_fraction": round(frac, 4), "top_suspicious_rows": top_indices(s)}
        if frac >= self.review_frac:
            return H.review(self, verdict=f"{int(fl.sum())} feature-outlier rows", **det)
        return H.ok(self, verdict="no feature-outlier subpopulation", **det)

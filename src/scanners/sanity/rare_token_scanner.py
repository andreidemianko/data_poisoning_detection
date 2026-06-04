from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class RareTokenScanner(BaseScanner):
    """Adapter: backdoor-trigger token detector (data-level, text, reference-free)."""
    name = "Sanity: NLP backdoor trigger tokens"
    category = ScannerCategory.SANITY

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.data_level import text_columns, combined_text, trigger_scan
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        cols = text_columns(df)
        if not cols:
            return H.skip(self, "no free-text columns")
        texts = combined_text(df, cols)
        rows, tokens = trigger_scan(texts)
        det = {"text_columns": cols, "n_rows": len(texts), "trigger_rows": rows,
               "trigger_tokens": tokens[:20]}
        if tokens:
            return H.block(self, verdict=f"trigger token(s) {tokens[:5]} in {rows} rows", **det)
        return H.ok(self, verdict="no injected trigger tokens", **det)

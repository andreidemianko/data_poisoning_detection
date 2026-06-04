from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class CharsetScanner(BaseScanner):
    """Adapter: charset/homoglyph detector (data-level, text). Blocks on a clear hit."""
    name = "Sanity: charset / homoglyph (mixed-script words)"
    category = ScannerCategory.SANITY
    block_rows = 2
    block_frac = 0.001

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.data_level import text_columns, combined_text, charset_scan
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        cols = text_columns(df)
        if not cols:
            return H.skip(self, "no free-text columns")
        texts = combined_text(df, cols)
        n = len(texts)
        rows, examples = charset_scan(texts)
        det = {"text_columns": cols, "n_rows": n, "homoglyph_rows": rows, "examples": examples}
        if rows >= max(self.block_rows, self.block_frac * n):
            return H.block(self, verdict=f"homoglyph: mixed-script words in {rows} rows", **det)
        return H.ok(self, verdict="no homoglyphs", **det)

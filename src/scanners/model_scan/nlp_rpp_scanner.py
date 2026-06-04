from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class NLPRPPScanner(BaseScanner):
    """Adapter: RPP on the fine-tuned BERT (noise in input embeddings). Catches backdoor."""
    name = "Model (NLP): RPP on BERT (input-embedding noise)"
    category = ScannerCategory.MODEL
    review_frac = 0.02

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.nlp_model_level import rpp_scores_bert
            from src.detectors.data_level import outlier_flags, top_indices
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        try:
            s, label_col, tcols = rpp_scores_bert(context.model_path, df)
        except ValueError as exc:
            return H.skip(self, str(exc))
        except ImportError:
            return H.skip(self, "transformers/torch not installed: pip install transformers")
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "RPP(BERT) failed", exc)
        if len(s) < 20:
            return H.skip(self, f"too few rows ({len(s)})")
        fl = outlier_flags(s)
        frac = float(fl.mean())
        det = {"text_columns": tcols, "label_column": label_col, "n_rows": len(s),
               "flagged": int(fl.sum()), "flagged_fraction": round(frac, 4),
               "top_suspicious_rows": top_indices(s),
               "note": "RPP measures prediction stability (backdoor signal) -> REVIEW"}
        if frac >= self.review_frac:
            return H.review(self, verdict=f"{int(fl.sum())} abnormally-stable rows (backdoor-like)", **det)
        return H.ok(self, verdict="no abnormally-stable subpopulation", **det)

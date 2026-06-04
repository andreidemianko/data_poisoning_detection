from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class NLPSpectralScanner(BaseScanner):
    """Adapter: Spectral Signatures on fine-tuned BERT embeddings (NLP model-level)."""
    name = "Model (NLP): Spectral Signatures on BERT embeddings"
    category = ScannerCategory.MODEL
    review_frac = 0.02

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.nlp_model_level import representation
            from src.detectors.model_level import spectral_scores
            from src.detectors.data_level import outlier_flags, top_indices
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "deps import failed", exc)
        df = context.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return H.skip(self, "dataset not loaded")
        try:
            emb, y, label_col, tcols = representation(context.model_path, df)
        except ValueError as exc:
            return H.skip(self, str(exc))
        except ImportError:
            return H.skip(self, "transformers/torch not installed: pip install transformers")
        except Exception as exc:  # noqa: BLE001
            return H.fail(self, "BERT encode failed", exc)
        if len(emb) < 20:
            return H.skip(self, f"too few rows ({len(emb)})")
        s = spectral_scores(emb, y)
        fl = outlier_flags(s)
        frac = float(fl.mean())
        det = {"text_columns": tcols, "label_column": label_col, "n_rows": len(emb),
               "flagged": int(fl.sum()), "flagged_fraction": round(frac, 4),
               "top_suspicious_rows": top_indices(s),
               "note": "NLP model-level on BERT embeddings, uncalibrated -> REVIEW"}
        if frac >= self.review_frac:
            return H.review(self, verdict=f"{int(fl.sum())} spectral-outlier rows in BERT space", **det)
        return H.ok(self, verdict="no spectral outliers in BERT space", **det)

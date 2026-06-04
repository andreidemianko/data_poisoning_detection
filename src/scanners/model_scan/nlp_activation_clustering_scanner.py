from __future__ import annotations
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScannerCategory
from src.scanners import _helpers as H


@register_scanner
class NLPActivationClusteringScanner(BaseScanner):
    """Adapter: Activation Clustering on fine-tuned BERT embeddings (NLP model-level)."""
    name = "Model (NLP): Activation Clustering on BERT embeddings"
    category = ScannerCategory.MODEL
    sil_review = 0.45

    def run(self, context: ScanContext) -> ScanResult:
        try:
            import pandas as pd
            from src.detectors.nlp_model_level import representation
            from src.detectors.model_level import activation_clustering
            from src.detectors.data_level import top_indices
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
        flags, scores, meta = activation_clustering(emb, y)
        sil = meta.get("max_minority_silhouette", -1.0)
        det = {"text_columns": tcols, "label_column": label_col, "n_rows": len(emb),
               "flagged": int(flags.sum()), "top_suspicious_rows": top_indices(scores),
               "note": "NLP model-level on BERT embeddings, uncalibrated -> REVIEW", **meta}
        if sil >= self.sil_review:
            return H.review(self, verdict=f"tight minority cluster in BERT space (silhouette={sil})", **det)
        return H.ok(self, verdict="no tight minority cluster in BERT space", **det)

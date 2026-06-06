"""
End-to-end poisoning gate, mirroring dataset_guard/gate.py: it coordinates all
detectors in the registry and assembles ONE decision (ALLOW / REVIEW / BLOCK).

The host adapter (src/scanners/sanity/poison_scanner.py) calls
    PostTrainGate.from_env(project_root).scan(dataset=..., model_path=...)
and converts report.as_dict() into the pipeline's ScanResult — exactly like
DatasetGuardScanner does for dataset_guard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .detectors import nlp_model_level as nlp
from .loaders import load_state_dict, read_dataset
from .models import Finding, FindingStatus, PostTrainReport
from .registry import CHECKS, DEFAULT_CONFIG, ScanInput


class PostTrainGate:
    """Runs the full poisoning ensemble and aggregates a single decision."""

    def __init__(self, config: Optional[Dict[str, float]] = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "PostTrainGate":
        """Build with default configuration (parallels DatasetSecurityGate.from_env)."""
        return cls()

    # --------------------------------------------------------------------- #
    def scan(
        self,
        dataset: Any = None,
        dataset_path: Optional[str] = None,
        model_state: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        reference_model_path: Optional[str] = None,
        reference_model_state: Optional[Dict[str, Any]] = None,
        clean_data: Any = None,
        clean_data_path: Optional[str] = None,
    ) -> PostTrainReport:
        if dataset is None:
            if not dataset_path:
                raise ValueError("either dataset (DataFrame) or dataset_path is required")
            dataset = read_dataset(dataset_path)
        if clean_data is None and clean_data_path:
            clean_data = read_dataset(clean_data_path)

        model_state = self._resolve_state(model_state, model_path)
        # ОПОРА (режим 2): табличную .pt грузим в state_dict; BERT-папку оставляем путём
        reference_model_state = self._resolve_state(reference_model_state, reference_model_path)

        inp = ScanInput(dataset=dataset, model_state=model_state,
                        model_path=model_path, config=self.config,
                        reference_model_path=reference_model_path,
                        reference_model_state=reference_model_state,
                        clean_data=clean_data)

        findings: List[Finding] = []
        for check in CHECKS:
            try:
                findings.append(check(inp))
            except Exception as exc:  # noqa: BLE001  — детектор не должен ронять весь скан
                findings.append(Finding(getattr(check, "__name__", "check"), "unknown",
                                        FindingStatus.ERROR, "detector crashed",
                                        {"error_type": type(exc).__name__, "error": str(exc)}))

        has_ref_model = reference_model_state is not None or self._is_transformer(reference_model_path)
        has_clean = clean_data is not None
        metadata = {
            "n_rows": int(len(dataset)),
            "model_present": model_state is not None or self._is_transformer(model_path),
            "reference_model_present": has_ref_model,
            "clean_sample_present": has_clean,
            "clean_sample_rows": int(len(clean_data)) if has_clean else 0,
            "mode": ("2 (calibrated vs reference)" if (has_ref_model or has_clean)
                     else "1 (reference-free triage)"),
            "config": dict(self.config),
        }
        return PostTrainReport.from_findings(findings, metadata)

    # --------------------------------------------------------------------- #
    def _resolve_state(self, model_state, model_path) -> Optional[Dict[str, Any]]:
        """state_dict для табличных model-детекторов. BERT-папку НЕ грузим как
        табличную модель (её используют NLP-детекторы по пути)."""
        if model_state is not None:
            return model_state
        if model_path and not self._is_transformer(model_path):
            return load_state_dict(model_path)
        return None

    @staticmethod
    def _is_transformer(model_path) -> bool:
        if not model_path:
            return False
        try:
            return nlp.is_transformer_dir(nlp.resolve_model_dir(model_path))
        except Exception:
            return False

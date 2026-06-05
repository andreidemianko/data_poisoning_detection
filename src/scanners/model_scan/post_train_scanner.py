from __future__ import annotations

from typing import Any

from src.core.factory import register_scanner
from src.core.loaders import project_root
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory


@register_scanner
class PostTrainGuardScanner(BaseScanner):
    """
    Adapter scanner for the post_train_guard engine (post-train model-level ensemble).

    Like DatasetGuardScanner, this does not reimplement detector logic. It calls
    the PostTrainGate API (the post-train ensemble: Spectral, Activation Clustering, RPP) and
    converts its report into the host pipeline's ScanResult format. Plugging the
    entire ensemble into the project is exactly this one file.
    """

    name = "Post-Train Guard: model-level ensemble (SPECTRE / Activation Clustering / RPP)"
    category = ScannerCategory.MODEL

    # Опора для режима 2 (необязательная). Заполняется из CLI (см. src/cli.py),
    # без изменения ScanContext/ядра: прокидка через атрибуты адаптера.
    reference_model_path: str | None = None   # старая/принятая модель (model_diff)
    clean_data_path: str | None = None        # чистый сэмпл данных (калибровка)

    def run(self, context: ScanContext) -> ScanResult:
        try:
            from post_train_guard.gate import PostTrainGate
        except Exception as exc:  # noqa: BLE001
            return ScanResult(
                name=self.name, category=self.category, status=ScanStatus.FAILED, passed=False,
                details={"reason": "post_train_guard import failed",
                         "error_type": type(exc).__name__, "error": str(exc)},
            )

        try:
            gate = PostTrainGate.from_env(project_root())
            report = gate.scan(
                dataset=context.dataset,
                dataset_path=context.dataset_path,
                model_state=context.model_state,
                model_path=context.model_path,
                reference_model_path=self.reference_model_path,   # режим 2: старая модель
                clean_data_path=self.clean_data_path,             # режим 2: чистый сэмпл
            )
            payload = report.as_dict()
        except Exception as exc:  # noqa: BLE001
            return ScanResult(
                name=self.name, category=self.category, status=ScanStatus.FAILED, passed=False,
                details={"reason": "post_train_guard scan failed",
                         "error_type": type(exc).__name__, "error": str(exc),
                         "dataset_path": context.dataset_path},
            )

        self._print_breakdown(payload.get("findings", []))

        decision = str(payload.get("decision", "")).upper()
        counts = payload.get("finding_counts", {})

        if decision == "ALLOW":
            status, passed = ScanStatus.PASSED, True
        elif decision == "REVIEW":
            status, passed = ScanStatus.HAND_CHECK, False
        else:  # BLOCK
            status, passed = ScanStatus.FAILED, False

        return ScanResult(
            name=self.name, category=self.category, status=status, passed=passed,
            details={
                "decision": decision,
                "finding_counts": counts,
                "findings": self._compact(payload.get("findings", [])),
                "metadata": payload.get("metadata", {}),
            },
        )

    @staticmethod
    def _print_breakdown(findings: list[dict[str, Any]]) -> None:
        """Печать статуса каждого детектора в консоль (пайплайн печатает только общий)."""
        icon = {"passed": "✅", "review": "⚠️", "block": "❌", "skipped": "⏭️", "error": "⛔"}
        order = {"block": 0, "review": 1, "error": 2, "passed": 3, "skipped": 4}
        for f in sorted(findings, key=lambda x: order.get(x.get("status", ""), 9)):
            ic = icon.get(f.get("status", ""), "•")
            print(f"   {ic} {str(f.get('detector', '')):44s} {f.get('verdict', '')}")

    @staticmethod
    def _compact(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-detector verdicts for UI/reporting, without dumping huge score arrays."""
        compact: list[dict[str, Any]] = []
        for f in findings:
            det = f.get("details", {})
            compact.append({
                "detector": f.get("detector"),
                "category": f.get("category"),
                "status": f.get("status"),
                "verdict": f.get("verdict"),
                "flagged": det.get("flagged"),
                "flagged_fraction": det.get("flagged_fraction"),
                "top_suspicious_rows": det.get("top_suspicious_rows", [])[:20],
                "trigger_tokens": det.get("trigger_tokens"),
                "examples": det.get("examples"),
                "max_minority_silhouette": det.get("max_minority_silhouette"),
            })
        return compact

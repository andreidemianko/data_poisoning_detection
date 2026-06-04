from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.factory import register_scanner
from src.core.loaders import project_root
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory


@register_scanner
class DatasetGuardScanner(BaseScanner):
    """
    Adapter scanner for the dataset_guard monolith.

    This scanner does not reimplement dataset_guard logic. It calls the existing
    DatasetSecurityGate API and converts its report into the host pipeline's
    ScanResult format.
    """

    name = "Dataset Guard: monolith security scan"
    category = ScannerCategory.SANITY

    def run(self, context: ScanContext) -> ScanResult:
        try:
            from dataset_guard.gate import DatasetSecurityGate
        except Exception as exc:
            return ScanResult(
                name=self.name,
                category=self.category,
                status=ScanStatus.FAILED,
                passed=False,
                details={
                    "reason": "dataset_guard import failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        try:
            gate = DatasetSecurityGate.from_env(project_root())
            report = gate.scan_path(Path(context.dataset_path))
            payload = report.as_dict()
        except Exception as exc:
            return ScanResult(
                name=self.name,
                category=self.category,
                status=ScanStatus.FAILED,
                passed=False,
                details={
                    "reason": "dataset_guard scan failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "dataset_path": context.dataset_path,
                },
            )

        decision = str(payload.get("decision", "")).upper()
        risk_score = float(payload.get("risk_score", 0.0) or 0.0)
        finding_counts = payload.get("finding_counts", {})
        file_count = len(payload.get("files", []))
        error_count = sum(len(item.get("errors", [])) for item in payload.get("files", []))

        if decision == "ALLOW" and error_count == 0:
            status = ScanStatus.PASSED
            passed = True
        elif decision == "REVIEW" and error_count == 0:
            status = ScanStatus.HAND_CHECK
            passed = False
        else:
            status = ScanStatus.FAILED
            passed = False

        return ScanResult(
            name=self.name,
            category=self.category,
            status=status,
            passed=passed,
            details={
                "decision": decision,
                "risk_score": risk_score,
                "finding_counts": finding_counts,
                "file_count": file_count,
                "error_count": error_count,
                "dataset_guard_report": self._compact_report(payload),
            },
        )

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        """
        Keep enough dataset_guard output for UI/reporting without dumping huge reports.
        """

        files = report.get("files", [])

        compact_files: list[dict[str, Any]] = []

        for item in files[:20]:
            findings = item.get("findings", [])

            compact_files.append(
                {
                    "file": item.get("file"),
                    "decision": item.get("decision"),
                    "risk_score": item.get("risk_score"),
                    "rows": item.get("rows"),
                    "columns": item.get("columns"),
                    "finding_counts": item.get("finding_counts"),
                    "errors": item.get("errors", []),
                    "top_findings": findings[:20],
                    "suppressed_count": item.get("metadata", {})
                    .get("suppression", {})
                    .get("suppressed_count", 0),
                }
            )

        return {
            "decision": report.get("decision"),
            "risk_score": report.get("risk_score"),
            "finding_counts": report.get("finding_counts"),
            "files": compact_files,
            "metadata": {
                "file_count": len(files),
            },
        }
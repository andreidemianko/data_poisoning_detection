from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .archives import ArchiveError, ArchiveResolver
from .config import AppConfig
from .detectors.factory import build_detectors
from .models import (
    CollectionScanResult,
    Decision,
    FileScanResult,
    Finding,
)
from .policy import RiskPolicy
from .profilers.base import Profiler
from .profilers.factory import build_profilers
from .readers import DatasetReader
from .suppression import FindingSuppressor
from .text_scanner import TextScanner
from .utils import error_dict, json_safe


class DatasetSecurityGate:
    """
    End-to-end dataset security gate.

    This class coordinates:
        - dataset discovery;
        - dataset reading;
        - text security scanning;
        - dataframe profiling;
        - finding suppression;
        - risk scoring;
        - report assembly.
    """

    def __init__(
        self,
        config: AppConfig,
        reader: DatasetReader | None = None,
        text_scanner: TextScanner | None = None,
        policy: RiskPolicy | None = None,
        suppressor: FindingSuppressor | None = None,
        profilers: list[Profiler] | None = None,
        archive_resolver: ArchiveResolver | None = None,
    ) -> None:
        self.config = config
        self.reader = reader or DatasetReader(config.reader)

        detector_bundle = build_detectors(config)
        profiler_bundle = build_profilers(config)

        self.text_scanner = text_scanner or TextScanner.from_config(
            config=config,
            fast_detectors=detector_bundle.fast_text,
            pii_detectors=detector_bundle.slow_pii,
        )

        self.archive_resolver = archive_resolver or ArchiveResolver(config.zip)
        self.policy = policy or RiskPolicy(config.policy)
        self.suppressor = suppressor or FindingSuppressor()
        self.profilers = profilers or profiler_bundle.dataframe

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "DatasetSecurityGate":
        """
        Build a gate using environment-based configuration.
        """

        config = AppConfig.from_env(project_root=project_root)

        return cls(config=config)

    def scan_path(self, input_path: Path | str) -> CollectionScanResult:
        """
        Scan a single dataset file, directory, or supported archive.
        """

        try:
            resolved_input = self.archive_resolver.resolve(input_path)
        except ArchiveError as exc:
            failed_result = self._failed_file_result(
                path=Path(input_path),
                error=error_dict(exc, where="archive_resolver"),
            )

            return self._build_collection_result(
                file_results=[failed_result],
                input_path=input_path,
            )

        try:
            dataset_files = self.reader.discover(resolved_input.path)

            if not dataset_files:
                return CollectionScanResult(
                    decision=Decision.ALLOW,
                    risk_score=0.0,
                    files=[],
                    metadata={
                        "input_path": str(input_path),
                        "resolved_input_path": str(resolved_input.path),
                        "message": "No supported dataset files were found.",
                        "config": self.config.as_dict(),
                    },
                )

            file_results = [self.scan_file(dataset_file.path) for dataset_file in dataset_files]

            return self._build_collection_result(
                file_results=file_results,
                input_path=input_path,
            )

        finally:
            resolved_input.cleanup()

    def scan_file(self, path: Path | str) -> FileScanResult:
        """
        Scan one dataset file.
        """

        dataset_files = self.reader.discover(path)

        if not dataset_files:
            return self._failed_file_result(
                path=Path(path),
                error={
                    "type": "UnsupportedDatasetFormatError",
                    "message": "No supported dataset file was found at the provided path.",
                    "where": "discover",
                },
            )

        dataset_file = dataset_files[0]

        try:
            frame = self.reader.read(dataset_file)
        except Exception as exc:
            return self._failed_file_result(
                path=dataset_file.path,
                error=error_dict(exc, where="read"),
            )

        text_findings, limits = self.text_scanner.scan_frame(frame)

        profile_findings: list[Finding] = []
        for profiler in self.profilers:
            try:
                profile_findings.extend(profiler.scan_frame(frame))
            except Exception as exc:
                profile_findings.append(
                    Finding(
                        category=self._quality_category(),
                        subtype="PROFILER_RUNTIME_ERROR",
                        rule_id=f"{profiler.name}.runtime_error",
                        severity=self._warn_severity(),
                        message=f"Profiler failed to run: {profiler.name}.",
                        detector=profiler.name,
                        confidence=1.0,
                        metadata={
                            "error": error_dict(exc, where=f"profiler.{profiler.name}"),
                        },
                    )
                )

        raw_findings = [
            *text_findings,
            *profile_findings,
        ]

        findings, suppression_metadata = self.suppressor.apply_with_metadata(raw_findings)
        findings = self._attach_row_evidence(findings, frame)

        breakdown = self.policy.explain(findings)

        return FileScanResult(
            file=dataset_file,
            decision=breakdown.decision,
            risk_score=breakdown.score,
            rows=len(frame),
            columns=[str(column) for column in frame.columns],
            findings=findings,
            detectors=[
                *self.text_scanner.describe_detectors(),
                *[profiler.describe() for profiler in self.profilers],
            ],
            limits=limits,
            metadata={
                "risk_breakdown": breakdown.as_dict(),
                "suppression": suppression_metadata,
                "evidence": {
                    "row_evidence_enabled": True,
                    "max_cell_chars": 500,
                    "warning": (
                        "Reports may contain original dataset cell values. "
                        "Do not publish reports if datasets contain sensitive data."
                    ),
                },
            },
        )

    def _attach_row_evidence(
        self,
        findings: list[Finding],
        frame,
        *,
        max_cell_chars: int = 500,
    ) -> list[Finding]:
        """
        Attach source dataset row snippets to findings.

        The evidence row is stored in finding.metadata["evidence_row"].
        The exact finding column value is stored in finding.metadata["evidence_value"].
        """

        enriched: list[Finding] = []

        for finding in findings:
            if finding.row_index is None:
                enriched.append(finding)
                continue

            row_data = self._get_row_evidence(
                frame=frame,
                row_index=finding.row_index,
                max_cell_chars=max_cell_chars,
            )

            if row_data is None:
                enriched.append(finding)
                continue

            metadata = {
                **finding.metadata,
                "evidence_row": row_data,
            }

            if finding.column and finding.column in row_data:
                metadata["evidence_value"] = row_data[finding.column]

            enriched.append(
                replace(
                    finding,
                    metadata=metadata,
                )
            )

        return enriched

    def _get_row_evidence(
        self,
        *,
        frame,
        row_index: int | str,
        max_cell_chars: int,
    ) -> dict[str, Any] | None:
        """
        Return a JSON-safe clipped row dictionary for a dataframe row index.
        """

        try:
            row = frame.loc[row_index]
        except Exception:
            try:
                if isinstance(row_index, str) and row_index.isdigit():
                    row = frame.loc[int(row_index)]
                else:
                    return None
            except Exception:
                return None

        try:
            import pandas as pd

            if isinstance(row, pd.DataFrame):
                if row.empty:
                    return None
                row = row.iloc[0]
        except Exception:
            pass

        try:
            raw_dict = row.to_dict()
        except Exception:
            return None

        return {
            str(key): self._clip_report_value(value, max_cell_chars=max_cell_chars)
            for key, value in raw_dict.items()
        }

    @staticmethod
    def _clip_report_value(value: Any, *, max_cell_chars: int) -> Any:
        """
        Convert dataframe cell values into JSON-safe clipped values.
        """

        if value is None:
            return None

        try:
            import pandas as pd

            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, (int, float, bool)):
            return value

        safe_value = json_safe(value)
        text = str(safe_value)

        if len(text) <= max_cell_chars:
            return text

        return text[:max_cell_chars] + "...[truncated]"

    def _failed_file_result(self, path: Path, error: dict[str, Any]) -> FileScanResult:
        """
        Build a failed file result without crashing collection scanning.
        """

        from .models import DatasetFile

        dataset_file = DatasetFile(path=path)
        findings: list[Finding] = []
        breakdown = self.policy.explain(findings)

        return FileScanResult(
            file=dataset_file,
            decision=Decision.REVIEW,
            risk_score=max(breakdown.score, self.config.policy.review_threshold),
            findings=findings,
            rows=None,
            columns=[],
            detectors=[
                *self.text_scanner.describe_detectors(),
                *[profiler.describe() for profiler in self.profilers],
            ],
            limits=None,
            errors=[error],
            metadata={
                "risk_breakdown": {
                    **breakdown.as_dict(),
                    "decision": Decision.REVIEW.value,
                    "reasons": [
                        "The file could not be scanned successfully and requires review.",
                        *breakdown.reasons,
                    ],
                },
            },
        )

    def _build_collection_result(
        self,
        *,
        file_results: list[FileScanResult],
        input_path: Path | str,
    ) -> CollectionScanResult:
        """
        Aggregate file-level results into a collection-level result.
        """

        if not file_results:
            return CollectionScanResult(
                decision=Decision.ALLOW,
                risk_score=0.0,
                files=[],
                metadata={
                    "input_path": str(input_path),
                    "config": self.config.as_dict(),
                },
            )

        risk_score = max(file_result.risk_score for file_result in file_results)

        if any(file_result.decision == Decision.BLOCK for file_result in file_results):
            decision = Decision.BLOCK
        elif any(file_result.decision == Decision.REVIEW for file_result in file_results):
            decision = Decision.REVIEW
        else:
            decision = Decision.ALLOW

        return CollectionScanResult(
            decision=decision,
            risk_score=risk_score,
            files=file_results,
            metadata={
                "input_path": str(input_path),
                "file_count": len(file_results),
                "config": self.config.as_dict(),
            },
        )

    @staticmethod
    def _quality_category():
        """
        Lazy import helper to avoid expanding the top-level import block.
        """

        from .models import Category

        return Category.QUALITY

    @staticmethod
    def _warn_severity():
        """
        Lazy import helper to avoid expanding the top-level import block.
        """

        from .models import Severity

        return Severity.WARN
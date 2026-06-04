from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

class Severity(str, Enum):
    """
    Level of severity
    """

    INFO = "INFO"
    WARN = "WARN"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"

class Category(str, Enum):
    """
    Base normalise category
    It's value forward to json/csv
    """
    
    SQLI = "sqli"
    XSS = "xss"
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    SECRET = "secret"

    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    TEMPLATE_INJECTION = "template_injection"
    LDAP_INJECTION = "ldap_injection"

    DATA_POISONING = "data_poisoning"
    SCHEMA = "schema"
    QUALITY = "quality"
    MALWARE_PATTERN = "malware_pattern"

    UNKNOWN = "unknown"

class Decision(str, Enum):
    """
    Final decision
    """

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class DatasetFile:
    "Discription of dataset file"

    path: Path
    source: str = "local"

    @property
    def dataset(self) -> str:
        """
        Name of dataset file

        If file lies in the directory:
            datasets/case_001/train.csv -> case_001

        If file single:
            train.csv -> train
        """

        parent_name = self.path.parent.name

        if parent_name and parent_name not in {".", ""}:
            return parent_name

        return self.path.stem

    @property
    def version(self) -> str:
        """
        If version of dataset file.

        While take stem file:
            train_v1.csv -> train_v1

        Late change VersionResolver.
        """

        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": self.source,
            "dataset": self.dataset,
            "version": self.version,
            "suffix": self.suffix,
        }

@dataclass(frozen=True)
class Finding:
    """
    A universal finding produced by any detector.

    Attributes:
        category: Base category (pii, sqli, xss, prompt_injection)
        subtype: Refinement (e.g., EMAIL_ADDRESS, UNION_SELECT, SCRIPT_TAG)
        rule_id: Stable rule identifier
        detector: Name of the detection engine that identified the issue.
    """

    category: Category
    rule_id: str
    severity: Severity
    message: str
    detector: str

    subtype: str | None = None
    confidence: float = 1.0

    column: str | None = None
    row_index: int | str | None = None
    value_sha256: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_key(self) -> tuple[Any, ...]:
        """
        Deduplication key.
        We don't include message because it may vary.
        """

        return (
            self.category.value,
            self.subtype,
            self.rule_id,
            self.detector,
            self.column,
            self.row_index,
            self.value_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "subtype": self.subtype,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "detector": self.detector,
            "confidence": self.confidence,
            "column": self.column,
            "row_index": self.row_index,
            "value_sha256": self.value_sha256,
            "metadata": self.metadata,
        }

@dataclass(frozen=True)
class DetectorInfo:
    """
    Information about a detector.
    """

    name: str
    status: str
    version: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "version": self.version,
            "error": self.error,
            "metadata": self.metadata,
        }

@dataclass(frozen=True)
class ScanLimits:
    """
    Limits that were actually applied during the scan.

    Especially important for PII to see:
        - how many cells were actually checked;
        - how many were skipped due to limits;
        - which limits were enabled.
    """

    max_rows: int | None = None
    max_cells_per_column: int | None = None
    max_total_cells: int | None = None

    pii_min_value_length: int | None = None
    pii_max_value_length: int | None = None
    pii_max_cells_per_column: int | None = None
    pii_max_total_cells: int | None = None
    pii_total_checked: int = 0
    pii_skipped_by_limit: int = 0
    pii_skipped_by_column_filter: int = 0
    pii_skipped_by_value_filter: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_rows": self.max_rows,
            "max_cells_per_column": self.max_cells_per_column,
            "max_total_cells": self.max_total_cells,
            "pii_min_value_length": self.pii_min_value_length,
            "pii_max_value_length": self.pii_max_value_length,
            "pii_max_cells_per_column": self.pii_max_cells_per_column,
            "pii_max_total_cells": self.pii_max_total_cells,
            "pii_total_checked": self.pii_total_checked,
            "pii_skipped_by_limit": self.pii_skipped_by_limit,
            "pii_skipped_by_column_filter": self.pii_skipped_by_column_filter,
            "pii_skipped_by_value_filter": self.pii_skipped_by_value_filter,
        }

@dataclass(frozen=True)
class FileScanResult:
    """
    Final result of file scan.
    """

    file: DatasetFile
    decision: Decision
    risk_score: float
    findings: list[Finding]

    rows: int | None = None
    columns: list[str] = field(default_factory=list)

    detectors: list[DetectorInfo] = field(default_factory=list)
    limits: ScanLimits | None = None

    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finding_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for finding in self.findings:
            key = finding.category.value
            counts[key] = counts.get(key, 0) + 1

        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file.as_dict(),
            "decision": self.decision.value,
            "risk_score": self.risk_score,
            "rows": self.rows,
            "columns": self.columns,
            "finding_counts": self.finding_counts(),
            "findings": [finding.as_dict() for finding in self.findings],
            "detectors": [detector.as_dict() for detector in self.detectors],
            "limits": self.limits.as_dict() if self.limits else None,
            "errors": self.errors,
            "metadata": self.metadata,
        }

@dataclass(frozen=True)
class CollectionScanResult:
    """
    Final result of collection scan.
    """

    decision: Decision
    risk_score: float
    files: list[FileScanResult]

    metadata: dict[str, Any] = field(default_factory=dict)

    def finding_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for file_result in self.files:
            for category, count in file_result.finding_counts().items():
                counts[category] = counts.get(category, 0) + count

        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "risk_score": self.risk_score,
            "finding_counts": self.finding_counts(),
            "files": [file_result.as_dict() for file_result in self.files],
            "metadata": self.metadata,
        }


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """
    Delete duplicate findings.
    """

    seen: set[tuple[Any, ...]] = set()
    result: list[Finding] = []

    for finding in findings:
        key = finding.normalized_key()

        if key in seen:
            continue

        seen.add(key)
        result.append(finding)

    return result


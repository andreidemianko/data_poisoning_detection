from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import CollectionScanResult, FileScanResult, Finding
from .utils import ensure_parent_directory, json_safe


def write_json_report(report: CollectionScanResult | FileScanResult, output_path: Path | str) -> None:
    """
    Write a full scan report as JSON.
    """

    path = Path(output_path)
    ensure_parent_directory(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(report.as_dict()),
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_findings_csv(report: CollectionScanResult | FileScanResult, output_path: Path | str) -> None:
    """
    Write a flat CSV file with one row per finding.
    """

    path = Path(output_path)
    ensure_parent_directory(path)

    rows = flatten_findings(report)

    fieldnames = [
        "file_path",
        "dataset",
        "version",
        "decision",
        "risk_score",
        "category",
        "subtype",
        "rule_id",
        "severity",
        "detector",
        "confidence",
        "column",
        "row_index",
        "value_sha256",
        "message",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def flatten_findings(report: CollectionScanResult | FileScanResult) -> list[dict[str, Any]]:
    """
    Convert a report object into flat finding rows.
    """

    if isinstance(report, FileScanResult):
        return _flatten_file_findings(report)

    rows: list[dict[str, Any]] = []

    for file_result in report.files:
        rows.extend(_flatten_file_findings(file_result))

    return rows


def build_summary(report: CollectionScanResult | FileScanResult) -> dict[str, Any]:
    """
    Build a compact summary dictionary.
    """

    if isinstance(report, FileScanResult):
        return {
            "decision": report.decision.value,
            "risk_score": report.risk_score,
            "file_count": 1,
            "finding_counts": report.finding_counts(),
            "error_count": len(report.errors),
        }

    return {
        "decision": report.decision.value,
        "risk_score": report.risk_score,
        "file_count": len(report.files),
        "finding_counts": report.finding_counts(),
        "error_count": sum(len(file_result.errors) for file_result in report.files),
    }


def print_summary(report: CollectionScanResult | FileScanResult) -> None:
    """
    Print a compact human-readable scan summary.
    """

    summary = build_summary(report)

    print(f"Decision: {summary['decision']}")
    print(f"Risk score: {summary['risk_score']}")
    print(f"Files: {summary['file_count']}")
    print(f"Errors: {summary['error_count']}")

    finding_counts = summary.get("finding_counts") or {}

    if not finding_counts:
        print("Findings: none")
        return

    print("Findings:")

    for category, count in finding_counts.items():
        print(f"  - {category}: {count}")


def _flatten_file_findings(file_result: FileScanResult) -> list[dict[str, Any]]:
    """
    Flatten findings from a single file result.
    """

    rows: list[dict[str, Any]] = []

    for finding in file_result.findings:
        rows.append(_finding_row(file_result, finding))

    return rows


def _finding_row(file_result: FileScanResult, finding: Finding) -> dict[str, Any]:
    """
    Convert one Finding object into a CSV-friendly dictionary.
    """

    return {
        "file_path": str(file_result.file.path),
        "dataset": file_result.file.dataset,
        "version": file_result.file.version,
        "decision": file_result.decision.value,
        "risk_score": file_result.risk_score,
        "category": finding.category.value,
        "subtype": finding.subtype,
        "rule_id": finding.rule_id,
        "severity": finding.severity.value,
        "detector": finding.detector,
        "confidence": finding.confidence,
        "column": finding.column,
        "row_index": finding.row_index,
        "value_sha256": finding.value_sha256,
        "message": finding.message,
    }
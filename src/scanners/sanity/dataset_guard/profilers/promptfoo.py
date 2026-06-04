from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PromptfooConfig
from ..models import Category, DetectorInfo, Finding, Severity
from ..utils import safe_row_index, value_fingerprint
from .base import Profiler


PROMPTFOO_STATIC_ASSERTIONS: list[dict[str, Any]] = [
    {
        "type": "not-regex",
        "value": r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?\b",
        "metric": "prompt_ignore_previous_instructions",
    },
    {
        "type": "not-regex",
        "value": r"(?i)\b(?:reveal|show|print|dump|exfiltrate)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
        "metric": "prompt_system_prompt_leak",
    },
    {
        "type": "not-regex",
        "value": r"(?i)\b(?:bypass|disable|ignore|override)\s+(?:safety|policy|guardrails?|filters?|restrictions?)\b",
        "metric": "prompt_policy_bypass",
    },
    {
        "type": "not-regex",
        "value": r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(?:a\s+)?(?:root|admin|developer|system|jailbroken)\b",
        "metric": "prompt_role_override",
    },
    {
        "type": "not-regex",
        "value": r"(?i)\b(?:dan\s+mode|developer\s+mode|jailbreak|do\s+anything\s+now)\b",
        "metric": "prompt_jailbreak_mode",
    },
]


@dataclass(frozen=True)
class PromptfooCandidate:
    """
    A dataframe cell selected for Promptfoo evaluation.
    """

    text: str
    column: str
    row_index: int | str
    value_sha256: str


class PromptfooProfiler(Profiler):
    """
    Optional Promptfoo-based profiler for prompt-injection-like dataset values.

    This profiler runs Promptfoo once per sampled batch by generating a temporary
    config with precomputed provider outputs. It does not call an LLM provider.
    """

    name = "promptfoo"

    def __init__(self, config: PromptfooConfig | None = None) -> None:
        self.config = config or PromptfooConfig()

    def describe(self) -> DetectorInfo:
        """
        Return report-friendly profiler metadata.
        """

        command_path = shutil.which(self.config.command)

        return DetectorInfo(
            name=self.name,
            status="available" if command_path else "unavailable",
            error=None if command_path else f"Command not found: {self.config.command}",
            metadata={
                "command": self.config.command,
                "command_path": command_path,
                "timeout_seconds": self.config.timeout_seconds,
                "max_value_length": self.config.max_value_length,
                "max_cells_per_column": self.config.max_cells_per_column,
                "max_total_cells": self.config.max_total_cells,
                "assertion_count": len(PROMPTFOO_STATIC_ASSERTIONS),
            },
        )

    def scan_frame(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Run Promptfoo against sampled text cells.
        """

        if not shutil.which(self.config.command):
            return []

        candidates = self._collect_candidates(frame)

        if not candidates:
            return []

        try:
            return self._run_promptfoo(candidates)
        except Exception as exc:
            return [
                Finding(
                    category=Category.QUALITY,
                    subtype="PROMPTFOO_RUNTIME_ERROR",
                    rule_id="promptfoo.runtime_error",
                    severity=Severity.WARN,
                    message="Promptfoo profiler failed to run.",
                    detector=self.name,
                    confidence=1.0,
                    metadata={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
            ]

    def _collect_candidates(self, frame: pd.DataFrame) -> list[PromptfooCandidate]:
        """
        Collect sampled text cells for Promptfoo evaluation.
        """

        candidates: list[PromptfooCandidate] = []
        total_limit = self.config.max_total_cells

        for column in frame.columns:
            if total_limit > 0 and len(candidates) >= total_limit:
                break

            column_name = str(column)

            if self._is_blocklisted_column(column_name):
                continue

            if self.config.column_allowlist and not self._is_allowlisted_column(column_name):
                continue

            series = frame[column]

            if not self._is_text_series(series):
                continue

            checked_for_column = 0

            for row_index, value in series.items():
                if total_limit > 0 and len(candidates) >= total_limit:
                    break

                if self.config.max_cells_per_column > 0:
                    if checked_for_column >= self.config.max_cells_per_column:
                        break

                if not isinstance(value, str):
                    continue

                text = value.strip()

                if not text:
                    continue

                if len(text) > self.config.max_value_length:
                    continue

                candidates.append(
                    PromptfooCandidate(
                        text=text,
                        column=column_name,
                        row_index=safe_row_index(row_index),
                        value_sha256=value_fingerprint(text),
                    )
                )

                checked_for_column += 1

        return candidates

    def _run_promptfoo(self, candidates: list[PromptfooCandidate]) -> list[Finding]:
        """
        Generate a temporary Promptfoo config, execute Promptfoo, and parse JSON output.
        """

        with tempfile.TemporaryDirectory(prefix="dataset_guard_promptfoo_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            config_path = temp_dir / "promptfooconfig.json"
            output_path = temp_dir / "promptfoo_results.json"

            config_payload = self._build_promptfoo_config(candidates, output_path)
            config_path.write_text(
                json.dumps(config_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            command = [
                self.config.command,
                "eval",
                "-c",
                str(config_path),
                "--output",
                str(output_path),
            ]

            completed = subprocess.run(
                command,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )

            if completed.returncode not in {0, 1}:
                raise RuntimeError(
                    "Promptfoo command failed: "
                    f"exit_code={completed.returncode}; "
                    f"stderr={completed.stderr[-2000:]}"
                )

            if not output_path.exists():
                raise RuntimeError(
                    "Promptfoo did not produce JSON output. "
                    f"stdout={completed.stdout[-2000:]}; stderr={completed.stderr[-2000:]}"
                )

            data = json.loads(output_path.read_text(encoding="utf-8"))

            return self._parse_promptfoo_results(data)

    def _build_promptfoo_config(
        self,
        candidates: list[PromptfooCandidate],
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Build a Promptfoo config object.

        providerOutput is used so Promptfoo runs assertions directly against the
        cell text without calling a provider.
        """

        tests: list[dict[str, Any]] = []

        for index, candidate in enumerate(candidates):
            tests.append(
                {
                    "description": f"dataset cell {index}",
                    "vars": {
                        "text": candidate.text,
                    },
                    "providerOutput": candidate.text,
                    "metadata": {
                        "column": candidate.column,
                        "row_index": candidate.row_index,
                        "value_sha256": candidate.value_sha256,
                    },
                    "assert": PROMPTFOO_STATIC_ASSERTIONS,
                }
            )

        return {
            "description": "Dataset Guard prompt injection static scan",
            "prompts": ["{{text}}"],
            "providers": ["echo"],
            "outputPath": str(output_path),
            "tests": tests,
        }

    def _parse_promptfoo_results(self, data: dict[str, Any]) -> list[Finding]:
        """
        Parse Promptfoo JSON output into normalized findings.

        The parser is intentionally defensive because Promptfoo JSON structures
        may differ across versions and output formats.
        """

        outputs = self._extract_outputs(data)
        findings: list[Finding] = []

        for output in outputs:
            if self._is_successful_output(output):
                continue

            metadata = self._extract_metadata(output)
            failed_metrics = self._extract_failed_metrics(output)

            if not failed_metrics:
                failed_metrics = ["promptfoo_failed_assertion"]

            for metric in failed_metrics:
                findings.append(
                    Finding(
                        category=Category.PROMPT_INJECTION,
                        subtype=str(metric).upper(),
                        rule_id=f"promptfoo.{metric}",
                        severity=Severity.REVIEW,
                        message="Promptfoo assertion failed on dataset text value.",
                        detector=self.name,
                        confidence=0.85,
                        column=metadata.get("column"),
                        row_index=metadata.get("row_index"),
                        value_sha256=metadata.get("value_sha256"),
                        metadata={
                            "metric": metric,
                        },
                    )
                )

        return findings

    @staticmethod
    def _extract_outputs(data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract Promptfoo output records from a JSON report.
        """

        results = data.get("results")

        if isinstance(results, dict):
            outputs = results.get("outputs")
            if isinstance(outputs, list):
                return [item for item in outputs if isinstance(item, dict)]

        outputs = data.get("outputs")

        if isinstance(outputs, list):
            return [item for item in outputs if isinstance(item, dict)]

        return []

    @staticmethod
    def _is_successful_output(output: dict[str, Any]) -> bool:
        """
        Determine whether a Promptfoo output passed all assertions.
        """

        if "success" in output:
            return bool(output.get("success"))

        grading_result = output.get("gradingResult")

        if isinstance(grading_result, dict) and "pass" in grading_result:
            return bool(grading_result.get("pass"))

        return True

    @staticmethod
    def _extract_metadata(output: dict[str, Any]) -> dict[str, Any]:
        """
        Extract test metadata from possible Promptfoo output locations.
        """

        for key in ("metadata",):
            value = output.get(key)
            if isinstance(value, dict):
                return value

        test_case = output.get("testCase")
        if isinstance(test_case, dict):
            metadata = test_case.get("metadata")
            if isinstance(metadata, dict):
                return metadata

        test = output.get("test")
        if isinstance(test, dict):
            metadata = test.get("metadata")
            if isinstance(metadata, dict):
                return metadata

        return {}

    @staticmethod
    def _extract_failed_metrics(output: dict[str, Any]) -> list[str]:
        """
        Extract failed assertion metrics from Promptfoo output.
        """

        grading_result = output.get("gradingResult")

        if not isinstance(grading_result, dict):
            return []

        component_results = grading_result.get("componentResults")

        if not isinstance(component_results, list):
            return []

        metrics: list[str] = []

        for component in component_results:
            if not isinstance(component, dict):
                continue

            if component.get("pass") is True:
                continue

            assertion = component.get("assertion")

            if isinstance(assertion, dict):
                metric = assertion.get("metric")
                if metric:
                    metrics.append(str(metric))

        return metrics

    def _is_allowlisted_column(self, column_name: str) -> bool:
        """
        Check Promptfoo column allowlist.
        """

        allowlist = self.config.column_allowlist or []
        normalized = column_name.lower()

        return any(item.lower() in normalized for item in allowlist)

    def _is_blocklisted_column(self, column_name: str) -> bool:
        """
        Check Promptfoo column blocklist.
        """

        blocklist = self.config.column_blocklist or []
        normalized = column_name.lower()

        return any(item.lower() in normalized for item in blocklist)

    @staticmethod
    def _is_text_series(series: pd.Series) -> bool:
        """
        Return True for text-like pandas columns.
        """

        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
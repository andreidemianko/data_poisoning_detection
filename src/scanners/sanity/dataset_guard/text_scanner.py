from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import AppConfig, PiiConfig, TextScanConfig
from .detectors.base import Detector
from .models import DetectorInfo, Finding, ScanLimits, dedupe_findings
from .utils import (
    is_missing_value,
    is_probably_text_value,
    normalize_text_variants,
    safe_row_index,
    value_fingerprint,
)


PII_COLUMN_NAME_PATTERN = re.compile(
    r"(?:^|[_ .-])("
    r"email|e-mail|mail|"
    r"phone|mobile|tel|telephone|"
    r"name|first_name|last_name|fullname|full_name|"
    r"address|street|city|zip|postal|postcode|"
    r"user|username|login|account|customer|client|"
    r"ip|ipv4|ipv6|"
    r"ssn|sin|passport|tax|tin|iban|card|credit|"
    r"comment|message|text|body|description|note"
    r")(?:$|[_ .-])",
    re.IGNORECASE,
)

PII_VALUE_HINT_PATTERN = re.compile(
    r"(@|(?:\+?\d[\d\s().-]{7,}\d)|(?:\d{1,3}\.){3}\d{1,3})"
)


@dataclass
class TextScanStats:
    """
    Runtime counters for text scanning.

    These counters are useful for reporting and for debugging performance issues.
    """

    fast_total_checked: int = 0
    fast_skipped_by_limit: int = 0

    pii_total_checked: int = 0
    pii_skipped_by_limit: int = 0
    pii_skipped_by_column_filter: int = 0
    pii_skipped_by_value_filter: int = 0


class TextScanner:
    """
    DataFrame text scanner.

    Fast detectors are allowed to scan broadly within general text limits.
    Slow PII detectors, such as Presidio, are called only for candidate columns
    and only within strict value and cell-count limits.
    """

    def __init__(
        self,
        *,
        fast_detectors: list[Detector],
        pii_detectors: list[Detector] | None = None,
        text_config: TextScanConfig | None = None,
        pii_config: PiiConfig | None = None,
    ) -> None:
        self.fast_detectors = fast_detectors
        self.pii_detectors = pii_detectors or []
        self.text_config = text_config or TextScanConfig()
        self.pii_config = pii_config or PiiConfig()

    @classmethod
    def from_config(
        cls,
        *,
        config: AppConfig,
        fast_detectors: list[Detector],
        pii_detectors: list[Detector] | None = None,
    ) -> "TextScanner":
        """
        Build a TextScanner from the top-level application config.
        """

        return cls(
            fast_detectors=fast_detectors,
            pii_detectors=pii_detectors or [],
            text_config=config.text_scan,
            pii_config=config.pii,
        )

    def describe_detectors(self) -> list[DetectorInfo]:
        """
        Return metadata for all configured detectors.
        """

        detectors = [*self.fast_detectors, *self.pii_detectors]

        return [detector.describe() for detector in detectors]

    def scan_frame(self, frame: pd.DataFrame) -> tuple[list[Finding], ScanLimits]:
        """
        Scan text-like values in a DataFrame.

        Returns normalized findings and the limits/counters used during scan.
        """

        findings: list[Finding] = []
        stats = TextScanStats()

        fast_total_limit = self.text_config.max_total_cells
        pii_total_limit = self.pii_config.max_total_cells

        for column in frame.columns:
            series = frame[column]

            if not self._is_text_series(series):
                continue

            pii_candidate_column = self._is_pii_candidate_column(column, series)
            fast_checked_for_column = 0
            pii_checked_for_column = 0

            for row_index, value in series.items():
                if is_missing_value(value):
                    continue

                if not is_probably_text_value(value):
                    continue

                text = str(value).strip()

                if not text:
                    continue

                context = {
                    "column": str(column),
                    "row_index": safe_row_index(row_index),
                    "value_sha256": value_fingerprint(text),
                }

                if self._can_run_fast_scan(text, fast_checked_for_column, stats, fast_total_limit):
                    findings.extend(self._scan_fast(text, context))
                    fast_checked_for_column += 1
                    stats.fast_total_checked += 1
                else:
                    stats.fast_skipped_by_limit += 1

                if self._can_run_pii_scan(
                    text=text,
                    pii_candidate_column=pii_candidate_column,
                    pii_checked_for_column=pii_checked_for_column,
                    stats=stats,
                    pii_total_limit=pii_total_limit,
                ):
                    findings.extend(self._scan_pii(text, context))
                    pii_checked_for_column += 1
                    stats.pii_total_checked += 1

        return dedupe_findings(findings), self._build_limits(stats)

    def _scan_fast(self, text: str, context: dict[str, Any]) -> list[Finding]:
        """
        Run fast detectors against normalized text variants.
        """

        findings: list[Finding] = []

        variants = normalize_text_variants(
            text,
            normalize_unicode=self.text_config.normalize_unicode,
            normalize_html=self.text_config.normalize_html,
            normalize_url=self.text_config.normalize_url,
        )

        for variant in variants:
            variant_context = {
                **context,
                "normalized_variant": variant != text,
            }

            for detector in self.fast_detectors:
                if not detector.is_available():
                    continue

                findings.extend(detector.scan_text(variant, variant_context))

        return findings

    def _scan_pii(self, text: str, context: dict[str, Any]) -> list[Finding]:
        """
        Run slow PII detectors against the original text only.

        Presidio-like engines should not be run against multiple decoded variants
        unless there is a strong reason to do so.
        """

        findings: list[Finding] = []

        for detector in self.pii_detectors:
            if not detector.is_available():
                continue

            findings.extend(detector.scan_text(text, context))

        return findings

    def _can_run_fast_scan(
        self,
        text: str,
        checked_for_column: int,
        stats: TextScanStats,
        total_limit: int,
    ) -> bool:
        """
        Check general fast-scan limits.
        """

        if total_limit > 0 and stats.fast_total_checked >= total_limit:
            return False

        if self.text_config.max_cells_per_column > 0:
            if checked_for_column >= self.text_config.max_cells_per_column:
                return False

        if len(text) > self.text_config.fast_max_value_length:
            return False

        return True

    def _can_run_pii_scan(
        self,
        *,
        text: str,
        pii_candidate_column: bool,
        pii_checked_for_column: int,
        stats: TextScanStats,
        pii_total_limit: int,
    ) -> bool:
        """
        Check PII-specific scan limits and candidate filters.
        """

        if not self.pii_detectors:
            return False

        if not pii_candidate_column and not self.pii_config.scan_all_text_columns:
            stats.pii_skipped_by_column_filter += 1
            return False

        if pii_total_limit > 0 and stats.pii_total_checked >= pii_total_limit:
            stats.pii_skipped_by_limit += 1
            return False

        if self.pii_config.max_cells_per_column > 0:
            if pii_checked_for_column >= self.pii_config.max_cells_per_column:
                stats.pii_skipped_by_limit += 1
                return False

        if not self._is_reasonable_pii_value(text):
            stats.pii_skipped_by_value_filter += 1
            return False

        return True

    def _is_reasonable_pii_value(self, text: str) -> bool:
        """
        Check whether a text value is suitable for slow PII scanning.
        """

        stripped = text.strip()

        if not stripped:
            return False

        length = len(stripped)

        if length < self.pii_config.min_value_length:
            return False

        if length > self.pii_config.max_value_length:
            return False

        if stripped.count("\n") > 30:
            return False

        return True

    def _is_pii_candidate_column(self, column: object, series: pd.Series) -> bool:
        """
        Decide whether a column is worth scanning with slow PII detectors.
        """

        column_name = str(column)

        if self._is_blocklisted_pii_column(column_name):
            return False

        if self.pii_config.scan_all_text_columns:
            return True

        if self._is_allowlisted_pii_column(column_name):
            return True

        if PII_COLUMN_NAME_PATTERN.search(column_name):
            return True

        if not self.pii_config.allow_hint_based_columns:
            return False

        return self._sample_has_pii_hint(series)

    def _is_allowlisted_pii_column(self, column_name: str) -> bool:
        """
        Check PII column allowlist.
        """

        allowlist = self.pii_config.column_allowlist or []

        if not allowlist:
            return False

        normalized = column_name.lower()

        return any(item.lower() in normalized for item in allowlist)

    def _is_blocklisted_pii_column(self, column_name: str) -> bool:
        """
        Check PII column blocklist.
        """

        blocklist = self.pii_config.column_blocklist or []

        if not blocklist:
            return False

        normalized = column_name.lower()

        return any(item.lower() in normalized for item in blocklist)

    def _sample_has_pii_hint(self, series: pd.Series) -> bool:
        """
        Check a small sample of values for obvious PII hints.

        This is a cheap prefilter to avoid running Presidio on unrelated columns.
        """

        checked = 0

        for value in series.dropna().head(self.pii_config.column_sample_size):
            if not isinstance(value, str):
                continue

            checked += 1

            if PII_VALUE_HINT_PATTERN.search(value):
                return True

        return False

    @staticmethod
    def _is_text_series(series: pd.Series) -> bool:
        """
        Return True for pandas columns that may contain text values.
        """

        return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

    def _build_limits(self, stats: TextScanStats) -> ScanLimits:
        """
        Build report-friendly scan limits and counters.
        """

        return ScanLimits(
            max_cells_per_column=self.text_config.max_cells_per_column,
            max_total_cells=self.text_config.max_total_cells,
            pii_min_value_length=self.pii_config.min_value_length,
            pii_max_value_length=self.pii_config.max_value_length,
            pii_max_cells_per_column=self.pii_config.max_cells_per_column,
            pii_max_total_cells=self.pii_config.max_total_cells,
            pii_total_checked=stats.pii_total_checked,
            pii_skipped_by_limit=stats.pii_skipped_by_limit,
            pii_skipped_by_column_filter=stats.pii_skipped_by_column_filter,
            pii_skipped_by_value_filter=stats.pii_skipped_by_value_filter,
        )
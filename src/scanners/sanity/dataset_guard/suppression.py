from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Category, Finding, dedupe_findings


@dataclass(frozen=True)
class SuppressionRuleResult:
    """
    Result of a suppression rule application.
    """

    kept: list[Finding]
    suppressed: list[Finding]
    reason: str


class FindingSuppressor:
    """
    Post-processing layer for detector findings.

    Detectors should stay simple and report what they see. Suppression rules
    remove known noisy overlaps after all detector results are available.
    """

    def apply(self, findings: list[Finding]) -> list[Finding]:
        """
        Apply all suppression rules and return cleaned findings.
        """

        cleaned = dedupe_findings(findings)
        cleaned = self._suppress_presidio_url_inside_email(cleaned)
        cleaned = dedupe_findings(cleaned)

        return cleaned

    def apply_with_metadata(self, findings: list[Finding]) -> tuple[list[Finding], dict[str, Any]]:
        """
        Apply suppression rules and return cleaned findings plus report metadata.
        """

        original_count = len(findings)
        cleaned = dedupe_findings(findings)

        suppressed_total: list[Finding] = []

        result = self._suppress_presidio_url_inside_email_with_result(cleaned)
        cleaned = result.kept
        suppressed_total.extend(result.suppressed)

        cleaned = dedupe_findings(cleaned)

        metadata = {
            "original_count": original_count,
            "final_count": len(cleaned),
            "suppressed_count": len(suppressed_total),
            "suppressed": [
                {
                    "category": finding.category.value,
                    "subtype": finding.subtype,
                    "rule_id": finding.rule_id,
                    "detector": finding.detector,
                    "column": finding.column,
                    "row_index": finding.row_index,
                    "value_sha256": finding.value_sha256,
                }
                for finding in suppressed_total
            ],
        }

        return cleaned, metadata

    def _suppress_presidio_url_inside_email(self, findings: list[Finding]) -> list[Finding]:
        """
        Suppress Presidio URL findings when Presidio also detected an email on
        the same cell.

        Presidio can sometimes split an email such as x@test.com into:
            - EMAIL_ADDRESS
            - URL

        For dataset security reports, EMAIL_ADDRESS is the more useful finding.
        """

        return self._suppress_presidio_url_inside_email_with_result(findings).kept

    def _suppress_presidio_url_inside_email_with_result(
        self,
        findings: list[Finding],
    ) -> SuppressionRuleResult:
        """
        Suppress noisy Presidio URL findings overlapping with email findings.
        """

        email_cells = {
            self._cell_key(finding)
            for finding in findings
            if finding.category == Category.PII
            and finding.detector == "presidio"
            and self._normalized_subtype(finding) == "EMAIL_ADDRESS"
        }

        kept: list[Finding] = []
        suppressed: list[Finding] = []

        for finding in findings:
            is_noisy_presidio_url = (
                finding.category == Category.PII
                and finding.detector == "presidio"
                and self._normalized_subtype(finding) == "URL"
                and self._cell_key(finding) in email_cells
            )

            if is_noisy_presidio_url:
                suppressed.append(finding)
                continue

            kept.append(finding)

        return SuppressionRuleResult(
            kept=kept,
            suppressed=suppressed,
            reason="Suppressed Presidio URL findings on cells where Presidio also detected EMAIL_ADDRESS.",
        )

    @staticmethod
    def _cell_key(finding: Finding) -> tuple[Any, ...]:
        """
        Return a key that identifies a scanned cell/value.
        """

        return (
            finding.column,
            finding.row_index,
            finding.value_sha256,
        )

    @staticmethod
    def _normalized_subtype(finding: Finding) -> str:
        """
        Return subtype normalized for comparisons.
        """

        return (finding.subtype or "").upper()
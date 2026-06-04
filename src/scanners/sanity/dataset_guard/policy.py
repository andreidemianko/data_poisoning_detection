from __future__ import annotations

from dataclasses import dataclass, field

from .config import PolicyConfig
from .models import Category, Decision, Finding, Severity
from .utils import clamp_float


DEFAULT_CATEGORY_WEIGHTS: dict[Category, float] = {
    Category.SECRET: 0.95,
    Category.PROMPT_INJECTION: 0.85,
    Category.SQLI: 0.75,
    Category.XSS: 0.75,
    Category.COMMAND_INJECTION: 0.75,
    Category.TEMPLATE_INJECTION: 0.70,
    Category.PATH_TRAVERSAL: 0.60,
    Category.LDAP_INJECTION: 0.60,
    Category.PII: 0.55,
    Category.DATA_POISONING: 0.65,
    Category.MALWARE_PATTERN: 0.90,
    Category.SCHEMA: 0.35,
    Category.QUALITY: 0.25,
    Category.UNKNOWN: 0.20,
}


DEFAULT_ALWAYS_BLOCK_CATEGORIES: set[Category] = {
    Category.SECRET,
    Category.MALWARE_PATTERN,
}


@dataclass(frozen=True)
class RiskBreakdown:
    """
    Human-readable explanation of risk scoring.

    This object is useful for reports and debugging why a dataset was allowed,
    sent to review, or blocked.
    """

    score: float
    decision: Decision
    finding_count: int
    max_severity: Severity | None
    category_counts: dict[str, int]
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "decision": self.decision.value,
            "finding_count": self.finding_count,
            "max_severity": self.max_severity.value if self.max_severity else None,
            "category_counts": self.category_counts,
            "reasons": self.reasons,
        }


class RiskPolicy:
    """
    Risk scoring and final decision policy.

    The policy converts normalized findings into:
        - numeric risk_score in [0.0, 1.0]
        - final decision: ALLOW, REVIEW, or BLOCK

    The scoring is intentionally simple and explainable.
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        category_weights: dict[Category, float] | None = None,
        always_block_categories: set[Category] | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS
        self.always_block_categories = always_block_categories or DEFAULT_ALWAYS_BLOCK_CATEGORIES

    def score(self, findings: list[Finding]) -> float:
        """
        Calculate a normalized risk score from findings.
        """

        if not findings:
            return 0.0

        raw_score = 0.0

        for finding in findings:
            weight = self.category_weights.get(finding.category, self.category_weights[Category.UNKNOWN])
            severity_multiplier = self._severity_multiplier(finding.severity)
            confidence = clamp_float(finding.confidence, 0.0, 1.0)

            # Repeated findings should increase risk, but not linearly without bound.
            raw_score += weight * severity_multiplier * confidence * 0.25

        return round(clamp_float(raw_score, 0.0, self.config.max_risk_score), 4)

    def decide(self, findings: list[Finding]) -> Decision:
        """
        Return the final decision for a set of findings.
        """

        breakdown = self.explain(findings)

        return breakdown.decision

    def explain(self, findings: list[Finding]) -> RiskBreakdown:
        """
        Return risk score, decision, and human-readable reasons.
        """

        score = self.score(findings)
        category_counts = self._category_counts(findings)
        max_severity = self._max_severity(findings)

        reasons: list[str] = []

        if not findings:
            return RiskBreakdown(
                score=0.0,
                decision=Decision.ALLOW,
                finding_count=0,
                max_severity=None,
                category_counts={},
                reasons=["No findings were detected."],
            )

        if any(finding.category in self.always_block_categories for finding in findings):
            blocked_categories = sorted(
                {finding.category.value for finding in findings if finding.category in self.always_block_categories}
            )
            reasons.append(f"Always-block category detected: {', '.join(blocked_categories)}.")

            return RiskBreakdown(
                score=max(score, self.config.block_threshold),
                decision=Decision.BLOCK,
                finding_count=len(findings),
                max_severity=max_severity,
                category_counts=category_counts,
                reasons=reasons,
            )

        if any(finding.severity == Severity.BLOCK for finding in findings):
            reasons.append("At least one finding has BLOCK severity.")

            return RiskBreakdown(
                score=max(score, self.config.block_threshold),
                decision=Decision.BLOCK,
                finding_count=len(findings),
                max_severity=max_severity,
                category_counts=category_counts,
                reasons=reasons,
            )

        if score >= self.config.block_threshold:
            reasons.append(f"Risk score {score} is greater than or equal to block threshold.")
            decision = Decision.BLOCK
        elif score >= self.config.review_threshold:
            reasons.append(f"Risk score {score} is greater than or equal to review threshold.")
            decision = Decision.REVIEW
        else:
            reasons.append(f"Risk score {score} is below review threshold.")
            decision = Decision.ALLOW

        return RiskBreakdown(
            score=score,
            decision=decision,
            finding_count=len(findings),
            max_severity=max_severity,
            category_counts=category_counts,
            reasons=reasons,
        )

    @staticmethod
    def _severity_multiplier(severity: Severity) -> float:
        """
        Convert finding severity into a numeric multiplier.
        """

        if severity == Severity.BLOCK:
            return 1.0

        if severity == Severity.REVIEW:
            return 0.7

        if severity == Severity.WARN:
            return 0.4

        return 0.2

    @staticmethod
    def _category_counts(findings: list[Finding]) -> dict[str, int]:
        """
        Count findings by normalized category.
        """

        counts: dict[str, int] = {}

        for finding in findings:
            category = finding.category.value
            counts[category] = counts.get(category, 0) + 1

        return dict(sorted(counts.items()))

    @staticmethod
    def _max_severity(findings: list[Finding]) -> Severity | None:
        """
        Return the highest severity found.
        """

        if not findings:
            return None

        order = {
            Severity.INFO: 0,
            Severity.WARN: 1,
            Severity.REVIEW: 2,
            Severity.BLOCK: 3,
        }

        return max((finding.severity for finding in findings), key=lambda item: order[item])
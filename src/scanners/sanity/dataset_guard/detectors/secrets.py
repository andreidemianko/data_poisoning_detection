from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Pattern

from ..models import Category, DetectorInfo, Finding, Severity
from .base import Detector, detector_status, finding_context_fields


@dataclass(frozen=True)
class SecretRule:
    rule_id: str
    subtype: str
    pattern: Pattern[str]
    message: str
    severity: Severity
    confidence: float


def _compile(pattern: str, flags: int = re.IGNORECASE) -> Pattern[str]:
    return re.compile(pattern, flags)


SECRET_RULES: list[SecretRule] = [
    SecretRule(
        rule_id="secret.aws_access_key",
        subtype="AWS_ACCESS_KEY",
        pattern=_compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", re.IGNORECASE),
        message="Possible AWS access key detected.",
        severity=Severity.BLOCK,
        confidence=0.95,
    ),
    SecretRule(
        rule_id="secret.github_token",
        subtype="GITHUB_TOKEN",
        pattern=_compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
        message="Possible GitHub personal access token detected.",
        severity=Severity.BLOCK,
        confidence=0.95,
    ),
    SecretRule(
        rule_id="secret.github_fine_grained_token",
        subtype="GITHUB_TOKEN",
        pattern=_compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
        message="Possible GitHub fine-grained token detected.",
        severity=Severity.BLOCK,
        confidence=0.95,
    ),
    SecretRule(
        rule_id="secret.slack_token",
        subtype="SLACK_TOKEN",
        pattern=_compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        message="Possible Slack token detected.",
        severity=Severity.BLOCK,
        confidence=0.95,
    ),
    SecretRule(
        rule_id="secret.stripe_live_key",
        subtype="STRIPE_SECRET_KEY",
        pattern=_compile(r"\bsk_live_[A-Za-z0-9]{16,}\b"),
        message="Possible Stripe live secret key detected.",
        severity=Severity.BLOCK,
        confidence=0.98,
    ),
    SecretRule(
        rule_id="secret.stripe_test_key",
        subtype="STRIPE_SECRET_KEY",
        pattern=_compile(r"\bsk_test_[A-Za-z0-9]{16,}\b"),
        message="Possible Stripe test secret key detected.",
        severity=Severity.REVIEW,
        confidence=0.9,
    ),
    SecretRule(
        rule_id="secret.private_key_block",
        subtype="PRIVATE_KEY_BLOCK",
        pattern=_compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |)?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
        message="Possible private key material detected.",
        severity=Severity.BLOCK,
        confidence=0.98,
    ),
    SecretRule(
        rule_id="secret.ssh_public_key",
        subtype="SSH_KEY_MATERIAL",
        pattern=_compile(r"\bssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/=]{40,}", re.IGNORECASE),
        message="Possible SSH key material detected.",
        severity=Severity.REVIEW,
        confidence=0.85,
    ),
    SecretRule(
        rule_id="secret.jwt_bearer",
        subtype="JWT_BEARER_TOKEN",
        pattern=_compile(
            r"\bAuthorization\s*:\s*Bearer\s+"
            r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
            re.IGNORECASE,
        ),
        message="Possible JWT bearer token detected.",
        severity=Severity.BLOCK,
        confidence=0.95,
    ),
    SecretRule(
        rule_id="secret.jwt",
        subtype="JWT_TOKEN",
        pattern=_compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
            re.IGNORECASE,
        ),
        message="Possible JWT token detected.",
        severity=Severity.REVIEW,
        confidence=0.9,
    ),
    SecretRule(
        rule_id="secret.generic_api_key_assignment",
        subtype="GENERIC_API_KEY",
        pattern=_compile(
            r"\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{16,}",
            re.IGNORECASE,
        ),
        message="Possible API key or secret assignment detected.",
        severity=Severity.REVIEW,
        confidence=0.75,
    ),
]


class SecretRegexDetector(Detector):
    """
    Fast regex detector for secrets and key material.

    This detector covers cloud credentials, CI/CD tokens, API keys, JWTs,
    and private-key fragments that should not be present in training datasets.
    """

    name = "secret_regex"

    def is_available(self) -> bool:
        return True

    def describe(self) -> DetectorInfo:
        return detector_status(
            name=self.name,
            available=True,
            metadata={
                "rule_count": len(SECRET_RULES),
                "entities": sorted({rule.subtype for rule in SECRET_RULES}),
            },
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        if not text:
            return []

        findings: list[Finding] = []
        context_fields = finding_context_fields(context)

        for rule in SECRET_RULES:
            match = rule.pattern.search(text)

            if not match:
                continue

            findings.append(
                Finding(
                    category=Category.SECRET,
                    subtype=rule.subtype,
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=rule.message,
                    detector=self.name,
                    confidence=rule.confidence,
                    metadata={
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "match_preview": self._preview(match.group(0)),
                    },
                    **context_fields,
                )
            )

        return findings

    @staticmethod
    def _preview(value: str) -> str:
        """
        Return a safe preview without exposing full secret value.
        """

        value = str(value)

        if len(value) <= 10:
            return "[redacted]"

        return value[:4] + "..." + value[-4:]
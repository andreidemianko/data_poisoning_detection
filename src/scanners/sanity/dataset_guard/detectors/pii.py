from __future__ import annotations

import re
from typing import Any

from ..config import PiiConfig
from ..models import Category, DetectorInfo, Finding, Severity
from ..utils import safe_optional_import
from .base import Detector, detector_status, finding_context_fields


class PiiRegexDetector(Detector):
    """
    Fast regex-based PII detector.

    This detector is intentionally lightweight and dependency-free. It is not as
    accurate as Presidio, but it provides a useful baseline for obvious PII such
    as emails, IP addresses, phone-like strings, and IBAN-like values.
    """

    name = "pii_regex"

    SYNTHETIC_EMAIL_DOMAINS = {
        "example.com",
        "company.com",
        "test.com",
        "email.com",
    }

    EMAIL = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    IPV4 = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    )

    PHONE = re.compile(
        r"""
        (?<![\w#])
        (?:
            \+\d{1,3}[\s.-]?
            (?:\(?\d{2,4}\)?[\s.-]?){1,3}
            \d{3,4}
            |
            \(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}
            |
            \d{3}[\s.-]\d{3}[\s.-]\d{4}
        )
        (?!\w)
        """,
        re.VERBOSE,
    )

    IBAN = re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        re.IGNORECASE,
    )

    CREDIT_CARD_LIKE = re.compile(
        r"""
        (?<!\d)
        (?:\d[ -]?){13,19}
        (?!\d)
        """,
        re.VERBOSE,
    )

    def is_available(self) -> bool:
        return True

    def describe(self) -> DetectorInfo:
        return detector_status(
            name=self.name,
            available=True,
            metadata={
                "entities": [
                    "EMAIL_ADDRESS",
                    "IP_ADDRESS",
                    "PHONE_NUMBER",
                    "IBAN",
                    "CREDIT_CARD_LIKE",
                ]
            },
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        if not text:
            return []

        findings: list[Finding] = []
        context_fields = finding_context_fields(context)

        for match in self.EMAIL.finditer(text):
            email = match.group(0)
            domain = email.rsplit("@", 1)[-1].lower()

            if domain in self.SYNTHETIC_EMAIL_DOMAINS:
                continue

            findings.append(
                Finding(
                    category=Category.PII,
                    subtype="EMAIL_ADDRESS",
                    rule_id="pii.regex.email",
                    severity=Severity.REVIEW,
                    message="Possible email address.",
                    detector=self.name,
                    confidence=0.9,
                    **context_fields,
                )
            )
            break

        if self.IPV4.search(text):
            findings.append(
                Finding(
                    category=Category.PII,
                    subtype="IP_ADDRESS",
                    rule_id="pii.regex.ipv4",
                    severity=Severity.REVIEW,
                    message="Possible IPv4 address.",
                    detector=self.name,
                    confidence=0.75,
                    **context_fields,
                )
            )

        if self.PHONE.search(text):
            findings.append(
                Finding(
                    category=Category.PII,
                    subtype="PHONE_NUMBER",
                    rule_id="pii.regex.phone",
                    severity=Severity.REVIEW,
                    message="Possible phone number.",
                    detector=self.name,
                    confidence=0.65,
                    **context_fields,
                )
            )

        if self.IBAN.search(text):
            findings.append(
                Finding(
                    category=Category.PII,
                    subtype="IBAN",
                    rule_id="pii.regex.iban",
                    severity=Severity.REVIEW,
                    message="Possible IBAN.",
                    detector=self.name,
                    confidence=0.85,
                    **context_fields,
                )
            )

        if self._has_luhn_valid_card(text):
            findings.append(
                Finding(
                    category=Category.PII,
                    subtype="CREDIT_CARD",
                    rule_id="pii.regex.credit_card_luhn",
                    severity=Severity.BLOCK,
                    message="Possible credit card number passing Luhn check.",
                    detector=self.name,
                    confidence=0.9,
                    **context_fields,
                )
            )

        return findings

    def _has_luhn_valid_card(self, text: str) -> bool:
        """
        Return True if text contains a credit-card-like number passing Luhn check.
        """

        for match in self.CREDIT_CARD_LIKE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))

            if not 13 <= len(digits) <= 19:
                continue

            if self._luhn_check(digits):
                return True

        return False

    @staticmethod
    def _luhn_check(digits: str) -> bool:
        """
        Validate a digit string using the Luhn checksum.
        """

        total = 0
        reverse_digits = digits[::-1]

        for index, char in enumerate(reverse_digits):
            value = int(char)

            if index % 2 == 1:
                value *= 2
                if value > 9:
                    value -= 9

            total += value

        return total % 10 == 0


class PresidioDetector(Detector):
    """
    Presidio-based PII detector.

    This detector can be expensive because it may use NLP models internally.
    It should be called only after column-level and value-level prefilters.
    """

    name = "presidio"

    def __init__(
        self,
        config: PiiConfig | None = None,
        entities: list[str] | None = None,
        language: str = "en",
    ) -> None:
        self.config = config or PiiConfig()
        self.entities = entities
        self.language = language

        self._available = False
        self._error: str | None = None
        self._version: str | None = None
        self._analyzer: Any | None = None

        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize Presidio safely.

        Optional dependency failures should not break the whole scanner.
        """

        presidio_module, error = safe_optional_import("presidio_analyzer")

        if presidio_module is None:
            self._available = False
            self._error = error
            return

        try:
            analyzer_cls = getattr(presidio_module, "AnalyzerEngine")
            self._analyzer = analyzer_cls()
            self._version = getattr(presidio_module, "__version__", None)
            self._available = True
        except Exception as exc:
            self._available = False
            self._error = f"{type(exc).__name__}: {exc}"

    def is_available(self) -> bool:
        return self._available and self._analyzer is not None

    def describe(self) -> DetectorInfo:
        return detector_status(
            name=self.name,
            available=self.is_available(),
            version=self._version,
            error=self._error,
            metadata={
                "language": self.language,
                "entities": self.entities,
                "max_value_length": self.config.max_value_length,
            },
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        if not self.is_available():
            return []

        if not self._is_reasonable_value(text):
            return []

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=self.entities,
                language=self.language,
            )
        except Exception:
            return []

        findings: list[Finding] = []
        context_fields = finding_context_fields(context)

        for result in results:
            entity_type = getattr(result, "entity_type", "UNKNOWN")
            score = float(getattr(result, "score", 0.0))

            findings.append(
                Finding(
                    category=Category.PII,
                    subtype=str(entity_type),
                    rule_id=f"pii.presidio.{str(entity_type).lower()}",
                    severity=self._severity_for_entity(str(entity_type), score),
                    message=f"Possible PII entity detected by Presidio: {entity_type}.",
                    detector=self.name,
                    confidence=score,
                    metadata={
                        "start": getattr(result, "start", None),
                        "end": getattr(result, "end", None),
                    },
                    **context_fields,
                )
            )

        return findings

    def _is_reasonable_value(self, text: str) -> bool:
        """
        Check value-level limits before running Presidio.
        """

        if not isinstance(text, str):
            return False

        stripped = text.strip()

        if not stripped:
            return False

        length = len(stripped)

        if length < self.config.min_value_length:
            return False

        if length > self.config.max_value_length:
            return False

        # Very multiline values are usually documents, logs, stack traces, or blobs.
        # They should be handled by a separate document-level scanner.
        if stripped.count("\n") > 30:
            return False

        return True

    @staticmethod
    def _severity_for_entity(entity_type: str, score: float) -> Severity:
        """
        Map Presidio entity type and confidence score to scanner severity.

        In dataset scanning, common PII such as emails or names should normally
        trigger REVIEW, not BLOCK. Highly sensitive identifiers can still block.
        """

        normalized = entity_type.upper()

        high_risk_entities = {
            "CREDIT_CARD",
            "CRYPTO",
            "IBAN_CODE",
            "IBAN",
            "US_SSN",
            "US_ITIN",
            "US_PASSPORT",
            "US_DRIVER_LICENSE",
            "UK_NHS",
            "NRP",
        }

        if normalized in high_risk_entities and score >= 0.75:
            return Severity.BLOCK

        if score >= 0.50:
            return Severity.REVIEW

        return Severity.WARN
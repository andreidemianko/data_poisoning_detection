from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..models import Category, DetectorInfo, Finding, Severity
from .base import Detector, detector_status, finding_context_fields


CYRILLIC_TO_LATIN_CONFUSABLES: dict[str, str] = {
    "а": "a",
    "А": "A",
    "е": "e",
    "Е": "E",
    "о": "o",
    "О": "O",
    "р": "p",
    "Р": "P",
    "с": "c",
    "С": "C",
    "у": "y",
    "У": "Y",
    "х": "x",
    "Х": "X",
    "і": "i",
    "І": "I",
    "ј": "j",
    "Ј": "J",
    "ѕ": "s",
    "Ѕ": "S",
    "ѵ": "v",
    "Ѵ": "V",
    "ӏ": "l",
    "ԁ": "d",
    "Ԁ": "D",
    "ԛ": "q",
    "Ԛ": "Q",
    "ԝ": "w",
    "Ԝ": "W",
    "Ь": "b",
    "В": "B",
    "Н": "H",
    "К": "K",
    "М": "M",
    "Т": "T",
}


ASCII_LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04FF]{3,}")


class HomoglyphDetector(Detector):
    """
    Fast detector for homoglyph and mixed-script text poisoning.

    This detector is designed for English-like datasets where individual Latin
    words may be poisoned by replacing some letters with visually similar
    Cyrillic characters.
    """

    name = "homoglyph"

    def is_available(self) -> bool:
        return True

    def describe(self) -> DetectorInfo:
        return detector_status(
            name=self.name,
            available=True,
            metadata={
                "confusable_count": len(CYRILLIC_TO_LATIN_CONFUSABLES),
                "scripts": ["latin", "cyrillic"],
            },
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        if not text:
            return []

        findings: list[Finding] = []
        context_fields = finding_context_fields(context)

        mixed_tokens = self._find_mixed_script_tokens(text)

        if mixed_tokens:
            findings.append(
                Finding(
                    category=Category.DATA_POISONING,
                    subtype="HOMOGLYPH_MIXED_SCRIPT_TOKEN",
                    rule_id="homoglyph.mixed_script_token",
                    severity=Severity.REVIEW,
                    message="Possible homoglyph poisoning: Latin-looking word contains Cyrillic confusable letters.",
                    detector=self.name,
                    confidence=self._confidence_for_tokens(mixed_tokens),
                    metadata={
                        "tokens": mixed_tokens[:10],
                        "token_count": len(mixed_tokens),
                    },
                    **context_fields,
                )
            )

        invisible_chars = self._find_invisible_unicode(text)

        if invisible_chars:
            findings.append(
                Finding(
                    category=Category.DATA_POISONING,
                    subtype="INVISIBLE_UNICODE_CONTROL",
                    rule_id="homoglyph.invisible_unicode_control",
                    severity=Severity.REVIEW,
                    message="Possible text poisoning: invisible Unicode control characters detected.",
                    detector=self.name,
                    confidence=0.85,
                    metadata={
                        "characters": invisible_chars[:10],
                        "character_count": len(invisible_chars),
                    },
                    **context_fields,
                )
            )

        return findings

    def _find_mixed_script_tokens(self, text: str) -> list[dict[str, Any]]:
        """
        Find tokens that mix ASCII Latin and Cyrillic confusable characters.
        """

        suspicious: list[dict[str, Any]] = []

        for match in TOKEN_RE.finditer(text):
            token = match.group(0)

            if not self._is_mixed_latin_cyrillic_token(token):
                continue

            confusables = [
                {
                    "char": char,
                    "looks_like": CYRILLIC_TO_LATIN_CONFUSABLES[char],
                    "unicode_name": self._unicode_name(char),
                }
                for char in token
                if char in CYRILLIC_TO_LATIN_CONFUSABLES
            ]

            if not confusables:
                continue

            suspicious.append(
                {
                    "token": token,
                    "normalized_token": self._normalize_confusables(token),
                    "start": match.start(),
                    "end": match.end(),
                    "confusables": confusables,
                }
            )

        return suspicious

    @staticmethod
    def _is_mixed_latin_cyrillic_token(token: str) -> bool:
        """
        Return True if a token contains ASCII Latin and Cyrillic confusables.
        """

        has_latin = bool(ASCII_LATIN_RE.search(token))
        has_cyrillic = bool(CYRILLIC_RE.search(token))
        has_confusable = any(char in CYRILLIC_TO_LATIN_CONFUSABLES for char in token)

        return has_latin and has_cyrillic and has_confusable

    @staticmethod
    def _normalize_confusables(token: str) -> str:
        """
        Replace known Cyrillic confusables with Latin lookalikes.
        """

        return "".join(CYRILLIC_TO_LATIN_CONFUSABLES.get(char, char) for char in token)

    @staticmethod
    def _find_invisible_unicode(text: str) -> list[dict[str, Any]]:
        """
        Find invisible Unicode format/control characters often used in poisoning.
        """

        suspicious: list[dict[str, Any]] = []

        for index, char in enumerate(text):
            category = unicodedata.category(char)

            if category not in {"Cf", "Cc"}:
                continue

            if char in {"\n", "\r", "\t"}:
                continue

            suspicious.append(
                {
                    "char": repr(char),
                    "index": index,
                    "unicode_name": HomoglyphDetector._unicode_name(char),
                    "category": category,
                }
            )

        return suspicious

    @staticmethod
    def _unicode_name(char: str) -> str:
        """
        Return a safe Unicode character name.
        """

        try:
            return unicodedata.name(char)
        except ValueError:
            return "UNKNOWN"

    @staticmethod
    def _confidence_for_tokens(tokens: list[dict[str, Any]]) -> float:
        """
        Estimate confidence from the number of suspicious tokens.
        """

        if len(tokens) >= 5:
            return 0.95

        if len(tokens) >= 2:
            return 0.9

        return 0.8
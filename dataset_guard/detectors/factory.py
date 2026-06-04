from __future__ import annotations

from dataclasses import dataclass
from .secrets import SecretRegexDetector
from ..config import AppConfig
from .base import Detector
from .pii import PiiRegexDetector, PresidioDetector
from .regex import RegexDetector
from .homoglyph import HomoglyphDetector


@dataclass(frozen=True)
class DetectorBundle:
    """
    A grouped detector set used by scanner components.

    fast_text:
        Cheap text detectors that can run broadly on text-like cells.

    slow_pii:
        Expensive PII detectors. These should be called only after column-level
        and value-level prefilters.
    """

    fast_text: list[Detector]
    slow_pii: list[Detector]


@dataclass(frozen=True)
class DetectorBundle:
    """
    A grouped detector set used by scanner components.
    """

    fast_text: list[Detector]
    slow_pii: list[Detector]


def build_detectors(config: AppConfig) -> DetectorBundle:
    """
    Build detector instances from application config.
    """

    fast_text: list[Detector] = []
    slow_pii: list[Detector] = []

    fast_text.append(RegexDetector())
    fast_text.append(SecretRegexDetector())
    fast_text.append(HomoglyphDetector())

    if config.engines.enable_pii_regex:
        fast_text.append(PiiRegexDetector())

    if config.engines.enable_presidio:
        slow_pii.append(PresidioDetector(config=config.pii))

    return DetectorBundle(
        fast_text=fast_text,
        slow_pii=slow_pii,
    )
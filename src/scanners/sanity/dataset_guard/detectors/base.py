from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import DetectorInfo, Finding

class Detector(ABC):
    """
    Base interface for all security detectors.

    A detector should be small, isolated, and safe to disable.
    Optional dependencies must not crash the whole scanner.
    """

    name: str = "detector"

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True when the detector is ready to scan.

        Optional detectors should return False when their dependency is missing
        or failed to initialize.
        """

        raise NotImplementedError

    @abstractmethod
    def describe(self) -> DetectorInfo:
        """
        Return report-friendly detector metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        """
        Scan a single text value and return normalized findings.

        The context may include:
            - column
            - row_index
            - value_sha256
            - dataset
            - file_path
            - normalized_variant
        """

        raise NotImplementedError

class FileDetector(ABC):
    """
    Base interface for detectors that scan files instead of cell text.

    Example:
        - gitleaks
        - full-file YARA
        - archive-level malware scanners
    """

    name: str = "file_detector"

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True when the detector is ready to scan files.
        """

        raise NotImplementedError

    @abstractmethod
    def describe(self) -> DetectorInfo:
        """
        Return report-friendly detector metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def scan_file(self, path: str, context: dict[str, Any] | None = None) -> list[Finding]:
        """
        Scan a file path and return normalized findings.
        """

        raise NotImplementedError

class NullDetector(Detector):
    """
    No-op text detector.

    Useful for tests and for degraded mode when an optional detector is disabled.
    """

    name = "null"

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def describe(self) -> DetectorInfo:
        return DetectorInfo(
            name=self.name,
            status="disabled",
            error=self.reason,
        )

    def scan_text(self, text: str, context: dict[str, Any] | None = None) -> list[Finding]:
        return []

class NullFileDetector(FileDetector):
    """
    No-op file detector.
    """

    name = "null_file"

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def describe(self) -> DetectorInfo:
        return DetectorInfo(
            name=self.name,
            status="disabled",
            error=self.reason,
        )

    def scan_file(self, path: str, context: dict[str, Any] | None = None) -> list[Finding]:
        return []

def context_value(context: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    """
    Safely read a value from detector context.
    """

    if context is None:
        return default

    return context.get(key, default)

def finding_context_fields(context: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extract common Finding fields from detector context.

    This keeps individual detectors small and consistent.
    """

    if context is None:
        return {
            "column": None,
            "row_index": None,
            "value_sha256": None,
        }

    return {
        "column": context.get("column"),
        "row_index": context.get("row_index"),
        "value_sha256": context.get("value_sha256"),
    }

def detector_status(
    *,
    name: str,
    available: bool,
    version: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DetectorInfo:
    """
    Build a consistent DetectorInfo object.
    """

    return DetectorInfo(
        name=name,
        status="available" if available else "unavailable",
        version=version,
        error=error,
        metadata=metadata or {},
    )

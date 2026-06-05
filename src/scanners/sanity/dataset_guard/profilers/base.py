from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..models import DetectorInfo, Finding


class Profiler(ABC):
    """
    Base interface for dataframe-level profilers.

    Profilers inspect the whole dataframe rather than individual text cells.
    They are useful for data quality, data poisoning, schema drift, and
    distribution-level checks.
    """

    name: str = "profiler"

    @abstractmethod
    def describe(self) -> DetectorInfo:
        """
        Return report-friendly profiler metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def scan_frame(self, frame: pd.DataFrame) -> list[Finding]:
        """
        Scan a dataframe and return normalized findings.
        """

        raise NotImplementedError
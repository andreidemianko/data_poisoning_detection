from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from .base import Profiler
from .poisoning import PoisoningProfiler

from .promptfoo import PromptfooProfiler


@dataclass(frozen=True)
class ProfilerBundle:
    """
    Grouped profiler set used by the dataset security gate.
    """

    dataframe: list[Profiler]


def build_profilers(config: AppConfig) -> ProfilerBundle:
    """
    Build dataframe-level profilers from application config.
    """

    dataframe_profilers: list[Profiler] = [
        PoisoningProfiler(config.poisoning),
    ]

    if config.engines.enable_promptfoo:
        dataframe_profilers.append(PromptfooProfiler(config.promptfoo))

    return ProfilerBundle(dataframe=dataframe_profilers)
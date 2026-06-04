from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type

from src.scanners.base import BaseScanner, ScannerCategory


class ScannerRegistry:
    def __init__(self) -> None:
        self._scanners: Dict[ScannerCategory, List[Type[BaseScanner]]] = {
            ScannerCategory.SANITY: [],
            ScannerCategory.STATS: [],
            ScannerCategory.MODEL: [],
        }

    def register(self, scanner_cls: Type[BaseScanner]) -> None:
        category = scanner_cls.category
        if scanner_cls not in self._scanners[category]:
            self._scanners[category].append(scanner_cls)

    def build(self, categories: Optional[Iterable[ScannerCategory]] = None) -> List[BaseScanner]:
        selected = categories or self._scanners.keys()
        instances: List[BaseScanner] = []
        for category in selected:
            for scanner_cls in self._scanners[category]:
                instances.append(scanner_cls())
        return instances


registry = ScannerRegistry()


def register_scanner(scanner_cls: Type[BaseScanner]) -> Type[BaseScanner]:
    registry.register(scanner_cls)
    return scanner_cls


def discover_scanners() -> None:
    scanners_dir = Path(__file__).resolve().parents[1] / "scanners"
    base_pkg = "src.scanners"

    for module in pkgutil.walk_packages([str(scanners_dir)], prefix=f"{base_pkg}."):
        if module.ispkg:
            continue

        importlib.import_module(module.name)
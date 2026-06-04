"""Маленькие фабрики ScanResult, чтобы адаптеры были тонкими. Не сканер (нет register)."""
from src.scanners.base import ScanResult, ScanStatus


def _r(scanner, status, passed, **details):
    return ScanResult(name=scanner.name, category=scanner.category,
                      status=status, passed=passed, details=details)


def skip(scanner, reason):           # проверка неприменима
    return _r(scanner, ScanStatus.SKIPPED, True, reason=reason)


def fail(scanner, reason, exc=None):  # ошибка выполнения
    d = {"reason": reason}
    if exc is not None:
        d.update(error_type=type(exc).__name__, error=str(exc))
    return _r(scanner, ScanStatus.FAILED, False, **d)


def ok(scanner, **details):           # чисто
    return _r(scanner, ScanStatus.PASSED, True, **details)


def review(scanner, **details):       # на ручную проверку
    return _r(scanner, ScanStatus.HAND_CHECK, False, **details)


def block(scanner, **details):        # блок (явная атака)
    return _r(scanner, ScanStatus.FAILED, False, **details)

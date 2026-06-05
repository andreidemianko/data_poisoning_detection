from __future__ import annotations

import hashlib
import hmac
import html
import importlib
import math
import os
import unicodedata
import urllib.parse
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

def safe_optional_import(module_name: str) -> tuple[ModuleType | None, str | None]:
    """
    Import an optional dependency without crashing the whole scanner.

    Some optional packages can fail with non-ImportError exceptions during import,
    especially when there are dependency or Python-version compatibility issues.
    """

    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

def is_missing_value(value: object) -> bool:
    """
    Return True if a scalar value should be treated as missing.

    This avoids calling pandas.isna() directly on list/dict/array-like values,
    because that can return non-scalar results and break boolean checks.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, float):
        return math.isnan(value)

    return False


def safe_row_index(row_index: object) -> int | str:
    """
    Convert a dataframe row index into a JSON-safe value.

    Integer indexes remain integers. Non-integer indexes are converted to strings.
    """

    if isinstance(row_index, int):
        return row_index

    return str(row_index)

def stable_text(value: object) -> str:
    """
    Convert a value into a stable text representation.

    This is used before hashing or text scanning. Bytes are decoded with
    replacement to avoid decode errors.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)

def get_fingerprint_key() -> bytes | None:
    """
    Read an optional HMAC key from the environment.

    If DATASET_SECURITY_HASH_KEY is set, value fingerprints become keyed HMACs.
    If it is not set, plain SHA-256 is used as a fallback.
    """

    raw_key = os.getenv("DATASET_SECURITY_HASH_KEY")

    if not raw_key:
        return None

    return raw_key.encode("utf-8", errors="replace")

def value_fingerprint(value: object, key: bytes | None = None) -> str:
    """
    Return a stable fingerprint for a value.

    Prefer keyed HMAC fingerprints for sensitive data. Plain SHA-256 is still
    supported for local/debug usage, but it is easier to brute-force for low
    cardinality values such as labels, booleans, or common emails.
    """

    text = stable_text(value).encode("utf-8", errors="replace")

    if key is None:
        key = get_fingerprint_key()

    if key:
        return hmac.new(key, text, hashlib.sha256).hexdigest()

    return hashlib.sha256(text).hexdigest()

def normalize_unicode_text(text: str) -> str:
    """
    Normalize Unicode text into a canonical compatibility form.
    """

    return unicodedata.normalize("NFKC", text)


def normalize_html_text(text: str) -> str:
    """
    Decode common HTML entities.
    """

    return html.unescape(text)

def normalize_url_text(text: str) -> str:
    """
    Decode percent-encoded URL sequences.

    The function is intentionally conservative: it decodes once only.
    """

    return urllib.parse.unquote(text)

def normalize_text_variants(value: object, *, normalize_unicode: bool = True, normalize_html: bool = True, normalize_url: bool = True,) -> list[str]:
    """
    Produce a small set of normalized text variants for security detection.

    The original text is always included first. Additional variants are added
    only when they differ from existing variants.
    """

    original = stable_text(value)
    variants: list[str] = []

    def add(candidate: str) -> None:
        if candidate not in variants:
            variants.append(candidate)

    add(original)

    current = original

    if normalize_unicode:
        current = normalize_unicode_text(current)
        add(current)

    if normalize_html:
        html_decoded = normalize_html_text(current)
        add(html_decoded)
    else:
        html_decoded = current

    if normalize_url:
        url_decoded = normalize_url_text(html_decoded)
        add(url_decoded)

    return variants


def is_probably_text_value(value: object) -> bool:
    """
    Return True if a value is suitable for text security scanning.
    """

    return isinstance(value, str) or isinstance(value, bytes)


def is_probably_text_value(value: object) -> bool:
    """
    Return True if a value is suitable for text security scanning.
    """

    return isinstance(value, str) or isinstance(value, bytes)

def text_length(value: object) -> int:
    """
    Return the length of the stable text representation.
    """

    return len(stable_text(value))

def clip_text(value: object, max_length: int) -> str:
    """
    Convert value to text and clip it to max_length characters.
    """

    text = stable_text(value)

    if max_length <= 0:
        return ""

    return text[:max_length]


def safe_lower(value: object) -> str:
    """
    Convert a value to lowercase text safely.
    """

    return stable_text(value).lower()


def contains_any_token(value: object, tokens: Iterable[str]) -> bool:
    """
    Check whether text contains any token from a list.
    """

    text = safe_lower(value)

    return any(token.lower() in text for token in tokens)

def json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.

    This is useful for metadata that may contain Path objects, sets, exceptions,
    bytes, or other non-serializable objects.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]

    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"

    return str(value)

def safe_ratio(numerator: float, denominator: float) -> float:
    """
    Return numerator / denominator, or 0.0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator

def clamp_float(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a float value into a closed interval.
    """

    return max(minimum, min(maximum, value))


def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)

def ensure_parent_directory(path: Path) -> None:
    """
    Create the parent directory for a file path if it does not exist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

def is_relative_to(child: Path, parent: Path) -> bool:
    """
    Backport-friendly Path.is_relative_to() helper.
    """

    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def has_supported_suffix(path: Path, suffixes: Iterable[str]) -> bool:
    """
    Check whether a path has one of the supported file suffixes.
    """

    allowed = {suffix.lower() for suffix in suffixes}

    return path.suffix.lower() in allowed

def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    """
    Return unique values while preserving the first-seen order.
    """

    seen: set[Any] = set()
    result: list[Any] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result

def error_dict(exc: BaseException, *, where: str | None = None) -> dict[str, Any]:
    """
    Convert an exception into a report-friendly dictionary.
    """

    data: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }

    if where:
        data["where"] = where

    return data

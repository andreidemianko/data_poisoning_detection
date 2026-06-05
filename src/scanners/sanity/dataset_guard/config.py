from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

def env_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean value from environment variables.

    Truthy values:
        1, true, yes, y, on

    Falsy values:
        0, false, no, n, off

    Any unknown value is treated as False unless default is used.
    """

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """
    Read an integer value from environment variables.

    If the variable is missing or invalid, default is returned.
    If minimum is provided, the returned value is clamped to that minimum.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value.strip())
    except ValueError:
        return default

    if minimum is not None:
        return max(value, minimum)

    return value

def env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    """
    Read a float value from environment variables.

    If the variable is missing or invalid, default is returned.
    Optional minimum and maximum bounds can be applied.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = float(raw_value.strip())
    except ValueError:
        return default

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value

def env_list(name: str, default: list[str] | None = None, separator: str = ",") -> list[str]:
    """
    Read a list of strings from an environment variable.

    Example:
        DATASET_SECURITY_PII_COLUMN_ALLOWLIST=email,phone,name
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default or []

    return [item.strip() for item in raw_value.split(separator) if item.strip()]

@dataclass(frozen=True)
class EngineConfig:
    """
    Engine enablement flags.

    Fast engines are intended to be cheap enough for normal runs.
    Slow engines must be explicitly enabled because they may require heavy
    dependencies or expensive NLP/model processing.
    """

    enable_yara: bool = True
    enable_injection_engine: bool = True
    enable_pii_regex: bool = True

    enable_slow_engines: bool = False
    enable_presidio: bool = False
    enable_promptfoo: bool = False
    enable_gitleaks: bool = False
    enable_great_expectations: bool = False
    enable_evidently: bool = False

    @classmethod
    def from_env(cls) -> "EngineConfig":
        enable_slow = env_bool("DATASET_SECURITY_ENABLE_SLOW_ENGINES", False)

        return cls(
            enable_yara=env_bool("DATASET_SECURITY_ENABLE_YARA", True),
            enable_injection_engine=env_bool("DATASET_SECURITY_ENABLE_INJECTION_ENGINE", True),
            enable_pii_regex=env_bool("DATASET_SECURITY_ENABLE_PII_REGEX", True),
            enable_slow_engines=enable_slow,
            enable_presidio=env_bool("DATASET_SECURITY_ENABLE_PRESIDIO", enable_slow),
            enable_promptfoo=env_bool("DATASET_SECURITY_ENABLE_PROMPTFOO", False),
            enable_gitleaks=env_bool("DATASET_SECURITY_ENABLE_GITLEAKS", enable_slow),
            enable_great_expectations=env_bool("DATASET_SECURITY_ENABLE_GREAT_EXPECTATIONS", enable_slow),
            enable_evidently=env_bool("DATASET_SECURITY_ENABLE_EVIDENTLY", enable_slow),
        )

@dataclass(frozen=True)
class PromptfooConfig:
    """
    Promptfoo integration configuration.

    Promptfoo is treated as an optional dataframe-level profiler. It should run
    on sampled candidate text values, not on every cell.
    """

    command: str = "promptfoo"

    timeout_seconds: int = 120

    max_value_length: int = 2048
    max_cells_per_column: int = 100
    max_total_cells: int = 1000

    column_allowlist: list[str] | None = None
    column_blocklist: list[str] | None = None

    @classmethod
    def from_env(cls) -> "PromptfooConfig":
        return cls(
            command=os.getenv("DATASET_SECURITY_PROMPTFOO_COMMAND", "promptfoo"),
            timeout_seconds=env_int("DATASET_SECURITY_PROMPTFOO_TIMEOUT_SECONDS", 120, minimum=1),
            max_value_length=env_int("DATASET_SECURITY_PROMPTFOO_MAX_VALUE_LENGTH", 2048, minimum=1),
            max_cells_per_column=env_int("DATASET_SECURITY_PROMPTFOO_MAX_CELLS_PER_COLUMN", 100, minimum=0),
            max_total_cells=env_int("DATASET_SECURITY_PROMPTFOO_MAX_TOTAL_CELLS", 1000, minimum=0),
            column_allowlist=env_list("DATASET_SECURITY_PROMPTFOO_COLUMN_ALLOWLIST"),
            column_blocklist=env_list("DATASET_SECURITY_PROMPTFOO_COLUMN_BLOCKLIST"),
        )

@dataclass(frozen=True)
class PiiConfig:
    """
    PII scanning configuration.

    Presidio should not be executed against every string-like cell.
    It should only run on candidate columns and within strict limits.
    """

    min_value_length: int = 3
    max_value_length: int = 2048

    max_cells_per_column: int = 1000
    max_total_cells: int = 10000

    column_sample_size: int = 50

    column_allowlist: list[str] | None = None
    column_blocklist: list[str] | None = None

    # If enabled, a column can be treated as PII-candidate when its sampled
    # values contain obvious hints such as emails, phone-like strings, or IPs.
    allow_hint_based_columns: bool = True

    # If enabled, Presidio may run on any text-like column.
    # This is disabled by default because it can be very slow.
    scan_all_text_columns: bool = False

    @classmethod
    def from_env(cls) -> "PiiConfig":
        return cls(
            min_value_length=env_int("DATASET_SECURITY_PII_MIN_VALUE_LENGTH", 3, minimum=0),
            max_value_length=env_int("DATASET_SECURITY_PII_MAX_VALUE_LENGTH", 2048, minimum=1),
            max_cells_per_column=env_int("DATASET_SECURITY_PII_MAX_CELLS_PER_COLUMN", 1000, minimum=0),
            max_total_cells=env_int("DATASET_SECURITY_PII_MAX_TOTAL_CELLS", 10000, minimum=0),
            column_sample_size=env_int("DATASET_SECURITY_PII_COLUMN_SAMPLE_SIZE", 50, minimum=1),
            column_allowlist=env_list("DATASET_SECURITY_PII_COLUMN_ALLOWLIST"),
            column_blocklist=env_list("DATASET_SECURITY_PII_COLUMN_BLOCKLIST"),
            allow_hint_based_columns=env_bool("DATASET_SECURITY_PII_ALLOW_HINT_BASED_COLUMNS", True),
            scan_all_text_columns=env_bool("DATASET_SECURITY_PII_SCAN_ALL_TEXT_COLUMNS", False),
        )

@dataclass(frozen=True)
class TextScanConfig:
    """
    General text scanning configuration.

    Fast detectors can scan more broadly than NLP-based detectors, but very
    large values still need limits to avoid pathological runtimes.
    """

    fast_max_value_length: int = 8192
    max_cells_per_column: int = 5000
    max_total_cells: int = 100000

    evidence_limit: int = 20

    normalize_unicode: bool = True
    normalize_html: bool = True
    normalize_url: bool = True

    @classmethod
    def from_env(cls) -> "TextScanConfig":
        return cls(
            fast_max_value_length=env_int("DATASET_SECURITY_FAST_MAX_VALUE_LENGTH", 8192, minimum=1),
            max_cells_per_column=env_int("DATASET_SECURITY_TEXT_MAX_CELLS_PER_COLUMN", 5000, minimum=0),
            max_total_cells=env_int("DATASET_SECURITY_TEXT_MAX_TOTAL_CELLS", 100000, minimum=0),
            evidence_limit=env_int("DATASET_SECURITY_EVIDENCE_LIMIT", 20, minimum=0),
            normalize_unicode=env_bool("DATASET_SECURITY_NORMALIZE_UNICODE", True),
            normalize_html=env_bool("DATASET_SECURITY_NORMALIZE_HTML", True),
            normalize_url=env_bool("DATASET_SECURITY_NORMALIZE_URL", True),
        )

@dataclass(frozen=True)
class ReaderConfig:
    """
    Dataset reader configuration.
    """

    supported_suffixes: tuple[str, ...] = (".csv", ".json", ".jsonl", ".parquet")

    csv_sample_rows: int | None = None
    json_sample_rows: int | None = None
    parquet_sample_rows: int | None = None

    @classmethod
    def from_env(cls) -> "ReaderConfig":
        csv_sample_rows = env_int("DATASET_SECURITY_CSV_SAMPLE_ROWS", 0, minimum=0)
        json_sample_rows = env_int("DATASET_SECURITY_JSON_SAMPLE_ROWS", 0, minimum=0)
        parquet_sample_rows = env_int("DATASET_SECURITY_PARQUET_SAMPLE_ROWS", 0, minimum=0)

        return cls(
            csv_sample_rows=csv_sample_rows or None,
            json_sample_rows=json_sample_rows or None,
            parquet_sample_rows=parquet_sample_rows or None,
        )

@dataclass(frozen=True)
class ZipConfig:
    """
    ZIP archive safety limits.

    These limits protect against:
    - zip slip paths;
    - too many files;
    - zip bombs;
    - oversized individual files.
    """

    max_files: int = 10000
    max_file_bytes: int = 200 * 1024 * 1024
    max_total_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_compression_ratio: float = 100.0

    @classmethod
    def from_env(cls) -> "ZipConfig":
        return cls(
            max_files=env_int("DATASET_SECURITY_ZIP_MAX_FILES", 10000, minimum=1),
            max_file_bytes=env_int("DATASET_SECURITY_ZIP_MAX_FILE_BYTES", 200 * 1024 * 1024, minimum=1),
            max_total_uncompressed_bytes=env_int(
                "DATASET_SECURITY_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES",
                2 * 1024 * 1024 * 1024,
                minimum=1,
            ),
            max_compression_ratio=env_float(
                "DATASET_SECURITY_ZIP_MAX_COMPRESSION_RATIO",
                100.0,
                minimum=1.0,
            ),
        )

@dataclass(frozen=True)
class PoisoningConfig:
    """
    Statistical data quality and data poisoning heuristic configuration.
    """

    min_rows_for_statistical_checks: int = 20

    class_imbalance_warn_ratio: float = 0.90
    class_imbalance_block_ratio: float = 0.98

    duplicate_rows_warn_ratio: float = 0.20
    duplicate_rows_block_ratio: float = 0.50

    constant_feature_warn_unique_values: int = 1

    rare_token_min_count: int = 3
    rare_token_label_correlation: float = 0.95

    numeric_outlier_zscore: float = 8.0

    @classmethod
    def from_env(cls) -> "PoisoningConfig":
        return cls(
            min_rows_for_statistical_checks=env_int(
                "DATASET_SECURITY_MIN_ROWS_FOR_STATISTICAL_CHECKS",
                20,
                minimum=1,
            ),
            class_imbalance_warn_ratio=env_float(
                "DATASET_SECURITY_CLASS_IMBALANCE_WARN_RATIO",
                0.90,
                minimum=0.0,
                maximum=1.0,
            ),
            class_imbalance_block_ratio=env_float(
                "DATASET_SECURITY_CLASS_IMBALANCE_BLOCK_RATIO",
                0.98,
                minimum=0.0,
                maximum=1.0,
            ),
            duplicate_rows_warn_ratio=env_float(
                "DATASET_SECURITY_DUPLICATE_ROWS_WARN_RATIO",
                0.20,
                minimum=0.0,
                maximum=1.0,
            ),
            duplicate_rows_block_ratio=env_float(
                "DATASET_SECURITY_DUPLICATE_ROWS_BLOCK_RATIO",
                0.50,
                minimum=0.0,
                maximum=1.0,
            ),
            numeric_outlier_zscore=env_float(
                "DATASET_SECURITY_NUMERIC_OUTLIER_ZSCORE",
                8.0,
                minimum=1.0,
            ),
        )

@dataclass(frozen=True)
class PolicyConfig:
    """
    Risk policy thresholds.
    """

    review_threshold: float = 0.35
    block_threshold: float = 0.70

    max_risk_score: float = 1.0

    @classmethod
    def from_env(cls) -> "PolicyConfig":
        return cls(
            review_threshold=env_float(
                "DATASET_SECURITY_REVIEW_THRESHOLD",
                0.35,
                minimum=0.0,
                maximum=1.0,
            ),
            block_threshold=env_float(
                "DATASET_SECURITY_BLOCK_THRESHOLD",
                0.70,
                minimum=0.0,
                maximum=1.0,
            ),
            max_risk_score=env_float(
                "DATASET_SECURITY_MAX_RISK_SCORE",
                1.0,
                minimum=0.0,
                maximum=1.0,
            ),

        )

@dataclass(frozen=True)
class AppConfig:
    """
    Top-level application configuration.
    """

    project_root: Path
    rules_path: Path

    engines: EngineConfig
    pii: PiiConfig
    text_scan: TextScanConfig
    promptfoo: PromptfooConfig
    reader: ReaderConfig
    zip: ZipConfig
    poisoning: PoisoningConfig
    policy: PolicyConfig

    allow_degraded: bool = False

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "AppConfig":
        if project_root is None:
            project_root = Path.cwd()

        rules_path_raw = os.getenv("DATASET_SECURITY_YARA_RULES_PATH")
        if rules_path_raw:
            rules_path = Path(rules_path_raw)
        else:
            rules_path = project_root / "rules" / "payloads.yar"

        return cls(
            project_root=project_root,
            rules_path=rules_path,
            engines=EngineConfig.from_env(),
            pii=PiiConfig.from_env(),
            text_scan=TextScanConfig.from_env(),
            reader=ReaderConfig.from_env(),
            promptfoo=PromptfooConfig.from_env(),
            zip=ZipConfig.from_env(),
            poisoning=PoisoningConfig.from_env(),
            policy=PolicyConfig.from_env(),
            allow_degraded=env_bool("DATASET_SECURITY_ALLOW_DEGRADED", False),
        )

    def as_dict(self) -> dict:
        return {
            "project_root": str(self.project_root),
            "rules_path": str(self.rules_path),
            "allow_degraded": self.allow_degraded,
            "engines": self.engines.__dict__,
            "pii": self.pii.__dict__,
            "promptfoo": self.promptfoo.__dict__,
            "text_scan": self.text_scan.__dict__,
            "reader": {
                "supported_suffixes": list(self.reader.supported_suffixes),
                "csv_sample_rows": self.reader.csv_sample_rows,
                "json_sample_rows": self.reader.json_sample_rows,
                "parquet_sample_rows": self.reader.parquet_sample_rows,
            },
            "zip": self.zip.__dict__,
            "poisoning": self.poisoning.__dict__,
            "policy": self.policy.__dict__,
        }
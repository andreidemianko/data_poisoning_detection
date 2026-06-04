from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ReaderConfig
from .models import DatasetFile
from .utils import has_supported_suffix


class DatasetReadError(RuntimeError):
    """
    Raised when a dataset file exists but cannot be read as a dataframe.
    """


class UnsupportedDatasetFormatError(ValueError):
    """
    Raised when the file suffix is not supported by the dataset reader.
    """


class DatasetReader:
    """
    Reader for supported tabular dataset formats.

    Supported formats:
        - CSV
        - JSON
        - JSONL
        - Parquet

    The reader returns pandas DataFrames because downstream profilers and
    scanners operate on tabular data.
    """

    def __init__(self, config: ReaderConfig | None = None) -> None:
        self.config = config or ReaderConfig()

    def read(self, dataset_file: DatasetFile | Path | str) -> pd.DataFrame:
        """
        Read a dataset file into a pandas DataFrame.
        """

        path = self._to_path(dataset_file)
        suffix = path.suffix.lower()

        if suffix not in self.config.supported_suffixes:
            raise UnsupportedDatasetFormatError(f"Unsupported dataset format: {suffix}")

        try:
            if suffix == ".csv":
                return self._read_csv(path)

            if suffix == ".json":
                return self._read_json(path)

            if suffix == ".jsonl":
                return self._read_jsonl(path)

            if suffix == ".parquet":
                return self._read_parquet(path)

        except Exception as exc:
            raise DatasetReadError(f"Failed to read dataset file {path}: {type(exc).__name__}: {exc}") from exc

        raise UnsupportedDatasetFormatError(f"Unsupported dataset format: {suffix}")

    def discover(self, input_path: Path | str) -> list[DatasetFile]:
        """
        Discover supported dataset files from a file or directory path.
        """

        path = Path(input_path)

        if path.is_file():
            if not has_supported_suffix(path, self.config.supported_suffixes):
                return []

            return [DatasetFile(path=path)]

        if not path.exists():
            return []

        if not path.is_dir():
            return []

        files: list[DatasetFile] = []

        for item in sorted(path.rglob("*")):
            if not item.is_file():
                continue

            if not has_supported_suffix(item, self.config.supported_suffixes):
                continue

            files.append(DatasetFile(path=item))

        return files

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """
        Read a CSV file.

        csv_sample_rows can be used to limit rows during development or CI.
        """

        return pd.read_csv(
            path,
            nrows=self.config.csv_sample_rows,
            low_memory=False,
        )

    def _read_json(self, path: Path) -> pd.DataFrame:
        """
        Read a JSON file.

        The method supports:
            - list of objects;
            - single object;
            - pandas-compatible JSON layouts.
        """

        with path.open("r", encoding="utf-8", errors="replace") as file:
            raw_text = file.read()

        data = json.loads(raw_text)

        if isinstance(data, list):
            frame = pd.DataFrame(data)
        elif isinstance(data, dict):
            frame = self._dataframe_from_dict(data)
        else:
            raise DatasetReadError(f"JSON root must be an object or a list, got {type(data).__name__}")

        if self.config.json_sample_rows is not None:
            frame = frame.head(self.config.json_sample_rows)

        return frame

    def _read_jsonl(self, path: Path) -> pd.DataFrame:
        """
        Read a JSON Lines file.

        Each non-empty line must be a valid JSON object.
        """

        rows: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DatasetReadError(f"Invalid JSONL at line {line_number}: {exc}") from exc

                if not isinstance(value, dict):
                    raise DatasetReadError(
                        f"JSONL line {line_number} must be an object, got {type(value).__name__}"
                    )

                rows.append(value)

                if self.config.json_sample_rows is not None:
                    if len(rows) >= self.config.json_sample_rows:
                        break

        return pd.DataFrame(rows)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """
        Read a Parquet file.

        parquet_sample_rows limits rows after loading. For very large Parquet
        files, this can be optimized later with pyarrow row-group reading.
        """

        frame = pd.read_parquet(path)

        if self.config.parquet_sample_rows is not None:
            frame = frame.head(self.config.parquet_sample_rows)

        return frame

    @staticmethod
    def _to_path(dataset_file: DatasetFile | Path | str) -> Path:
        """
        Convert supported input types to a Path object.
        """

        if isinstance(dataset_file, DatasetFile):
            return dataset_file.path

        return Path(dataset_file)

    @staticmethod
    def _dataframe_from_dict(data: dict[str, Any]) -> pd.DataFrame:
        """
        Convert a JSON object into a DataFrame.

        If all values are lists of the same length, pandas can build columns
        directly. Otherwise, the object is treated as a single-row record.
        """

        if not data:
            return pd.DataFrame([{}])

        values = list(data.values())

        if all(isinstance(value, list) for value in values):
            lengths = {len(value) for value in values}

            if len(lengths) == 1:
                return pd.DataFrame(data)

        return pd.DataFrame([data])
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import run_all_datasets_4_modes as scan_runner


TARGET_PRIORITY = (
    "label",
    "output",
    "intent",
    "category",
    "Class",
    "Risk",
    "Exited",
    "SeriousDlqin2yrs",
    "default.payment.next.month",
    "target",
    "y",
)
LABEL_LIKE = {value.lower() for value in TARGET_PRIORITY} | {"tags"}
TEXT_HINTS = {"text", "sentence", "input", "instruction", "response"}
DECISION_ORDER = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2, "MISSING": -1}


@dataclass
class ArtResult:
    dataset_name: str
    dataset_number: int
    dataset_path: str
    expected: str
    target: str | None
    n_rows: int
    n_used: int
    n_ref_used: int
    n_classes: int
    feature_mode: str
    issue_count: int | None
    issue_rate: float | None
    clean_reference_issue_rate: float | None
    excess_issue_rate: float | None
    threshold_quantile: float
    decision: str | None
    risk_score: float | None
    duration_sec: float
    top_issue_indices: str
    raw_report_path: str
    error: str | None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def normalize_decision(value: Any) -> str:
    text = str(value or "").upper()
    return text if text in {"ALLOW", "REVIEW", "BLOCK"} else "MISSING"


def stronger_decision(left: str | None, right: str | None) -> str:
    left = normalize_decision(left)
    right = normalize_decision(right)
    return left if DECISION_ORDER[left] >= DECISION_ORDER[right] else right


def expected_by_number(number: int, clean_number: int) -> str:
    return "clean" if number == clean_number else "poisoned"


def latest_profile_index(root: Path, mode: str) -> Path:
    profiles_root = root / "outputs" / "all_datasets_profiles"
    candidates = sorted(
        profiles_root.glob("*/index.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            rows = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if sum(1 for row in rows if row.get("mode") == mode) >= 40:
            return candidate
    raise FileNotFoundError(f"no full all_datasets_profiles index.json with mode={mode!r}")


def resolve_project_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def infer_target(df: pd.DataFrame, dataset_name: str) -> str | None:
    config = scan_runner.TABULAR_CONFIGS.get(dataset_name)
    if config and config.get("target") in df.columns:
        return str(config["target"])
    for candidate in TARGET_PRIORITY:
        if candidate in df.columns:
            return candidate
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported dataset format: {path.suffix}")


def choose_feature_columns(df: pd.DataFrame, target: str, drop: list[str], dataset_name: str) -> tuple[str, list[str]]:
    candidates = [col for col in df.columns if col != target and col not in set(drop)]
    object_cols = [col for col in candidates if df[col].dtype == object or str(df[col].dtype).startswith("string")]
    long_text_cols: list[str] = []
    for col in object_cols:
        if col.lower() in LABEL_LIKE and col.lower() not in TEXT_HINTS:
            continue
        series = df[col].dropna().astype(str)
        if col.lower() in TEXT_HINTS or (len(series) and float(series.str.len().mean()) >= 20):
            long_text_cols.append(col)

    if dataset_name in scan_runner.NLP_DATASETS and long_text_cols:
        return "text", long_text_cols
    if long_text_cols and not df[candidates].select_dtypes(include="number").shape[1]:
        return "text", long_text_cols
    return "tabular", candidates


def stratified_sample(df: pd.DataFrame, target: str, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    y = df[target]
    counts = y.value_counts(dropna=False)
    stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
    sample_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=max_rows,
        random_state=random_state,
        stratify=stratify,
    )
    return df.iloc[np.sort(sample_idx)].copy()


def make_text(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return df[feature_columns].fillna("").astype(str).agg(" ".join, axis=1).to_numpy()


def dense_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float32)
    return np.nan_to_num(array, copy=False)


def build_tabular_transformer(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_cols = list(df[feature_columns].select_dtypes(include="number").columns)
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers)


def fit_transform_features(
    candidate_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_mode: str,
    feature_columns: list[str],
    max_features: int,
    svd_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([reference_df[feature_columns], candidate_df[feature_columns]], axis=0, ignore_index=True)
    ref_n = len(reference_df)
    if feature_mode == "text":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, dtype=np.float32)
        combined_text = pd.concat(
            [
                pd.Series(make_text(reference_df, feature_columns)),
                pd.Series(make_text(candidate_df, feature_columns)),
            ],
            axis=0,
            ignore_index=True,
        )
        matrix = vectorizer.fit_transform(combined_text)
        if matrix.shape[1] >= 3 and svd_dims > 0:
            n_components = min(svd_dims, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if n_components >= 2:
                matrix = TruncatedSVD(n_components=n_components, random_state=13).fit_transform(matrix)
        matrix = dense_array(matrix)
    else:
        transformer = build_tabular_transformer(pd.concat([reference_df, candidate_df], axis=0, ignore_index=True), feature_columns)
        matrix = dense_array(transformer.fit_transform(combined))

    reference_x = matrix[:ref_n]
    candidate_x = matrix[ref_n:]
    return candidate_x, reference_x


def art_spectral_scores(matrix: np.ndarray) -> np.ndarray:
    from art.defences.detector.poison import SpectralSignatureDefense

    if matrix.shape[0] < 3:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    scores = SpectralSignatureDefense.spectral_signature_scores(matrix)
    return np.asarray(scores).reshape(-1)


def scores_by_class(x: np.ndarray, labels: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        result[str(label)] = (indices, art_spectral_scores(x[indices]))
    return result


def thresholds_from_reference(
    reference_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    threshold_quantile: float,
) -> tuple[dict[str, float], float]:
    all_scores = np.concatenate([scores for _indices, scores in reference_scores.values() if len(scores)]) if reference_scores else np.array([])
    global_threshold = float(np.quantile(all_scores, threshold_quantile)) if len(all_scores) else math.inf
    thresholds: dict[str, float] = {}
    for label, (_indices, scores) in reference_scores.items():
        if len(scores) >= 5:
            thresholds[label] = float(np.quantile(scores, threshold_quantile))
        else:
            thresholds[label] = global_threshold
    return thresholds, global_threshold


def count_issues(
    class_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: dict[str, float],
    global_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    issue_indices: list[int] = []
    issue_scores: list[float] = []
    for label, (indices, scores) in class_scores.items():
        threshold = thresholds.get(label, global_threshold)
        if not math.isfinite(threshold):
            continue
        mask = scores > threshold
        issue_indices.extend(indices[mask].tolist())
        issue_scores.extend(scores[mask].tolist())

    if not issue_indices:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    order = np.argsort(np.asarray(issue_scores))[::-1]
    return np.asarray(issue_indices, dtype=int)[order], np.asarray(issue_scores, dtype=np.float32)[order]


def decision_from_excess(excess: float, review_excess: float, block_excess: float) -> tuple[str, float]:
    risk = min(1.0, div(excess, block_excess))
    if excess >= block_excess:
        return "BLOCK", risk
    if excess >= review_excess:
        return "REVIEW", risk
    return "ALLOW", risk


def run_art_one(
    root: Path,
    index_row: dict[str, Any],
    raw_root: Path,
    clean_number: int,
    max_rows: int,
    max_features: int,
    svd_dims: int,
    threshold_quantile: float,
    review_excess: float,
    block_excess: float,
    random_state: int,
) -> ArtResult:
    started = time.perf_counter()
    dataset_name = str(index_row["dataset_name"])
    dataset_number = int(index_row["dataset_number"])
    expected = str(index_row.get("expected") or expected_by_number(dataset_number, clean_number))
    dataset_path = resolve_project_path(root, str(index_row.get("dataset_path") or ""))
    raw_path = raw_root / dataset_name / f"dataset_{dataset_number}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError(str(dataset_path or "missing dataset_path"))
        reference_path = dataset_path.parent / f"dataset_{clean_number}{dataset_path.suffix}"
        if not reference_path.exists():
            matches = sorted(dataset_path.parent.glob(f"dataset_{clean_number}.*"))
            if not matches:
                raise FileNotFoundError(f"missing clean reference for {dataset_name}")
            reference_path = matches[0]

        candidate_df = load_dataset(dataset_path)
        reference_df = load_dataset(reference_path)
        n_rows = len(candidate_df)
        target = infer_target(reference_df, dataset_name) or infer_target(candidate_df, dataset_name)
        if target is None or target not in candidate_df.columns or target not in reference_df.columns:
            raise ValueError("missing target column")

        drop = list(scan_runner.TABULAR_CONFIGS.get(dataset_name, {}).get("drop", []))
        candidate_df = candidate_df.dropna(subset=[target]).copy()
        reference_df = reference_df.dropna(subset=[target]).copy()
        candidate_df = stratified_sample(candidate_df, target, max_rows, random_state)
        reference_df = stratified_sample(reference_df, target, max_rows, random_state)

        combined_for_columns = pd.concat([reference_df, candidate_df], axis=0, ignore_index=True)
        feature_mode, feature_columns = choose_feature_columns(combined_for_columns, target, drop, dataset_name)
        if not feature_columns:
            raise ValueError("no usable feature columns")

        candidate_x, reference_x = fit_transform_features(
            candidate_df,
            reference_df,
            feature_mode,
            feature_columns,
            max_features=max_features,
            svd_dims=svd_dims,
        )
        candidate_labels = candidate_df[target].astype(str).to_numpy()
        reference_labels = reference_df[target].astype(str).to_numpy()
        if len(set(reference_labels.tolist())) < 2:
            raise ValueError("need at least two classes in clean reference")

        reference_scores = scores_by_class(reference_x, reference_labels)
        candidate_scores = scores_by_class(candidate_x, candidate_labels)
        thresholds, global_threshold = thresholds_from_reference(reference_scores, threshold_quantile)
        reference_issue_indices, _reference_issue_scores = count_issues(reference_scores, thresholds, global_threshold)
        candidate_issue_indices, candidate_issue_scores = count_issues(candidate_scores, thresholds, global_threshold)

        issue_count = int(len(candidate_issue_indices))
        reference_issue_count = int(len(reference_issue_indices))
        issue_rate = div(issue_count, len(candidate_df))
        reference_issue_rate = div(reference_issue_count, len(reference_df))
        excess = max(0.0, issue_rate - reference_issue_rate)
        decision, risk = decision_from_excess(excess, review_excess, block_excess)
        top_original_indices = [str(candidate_df.index[int(i)]) for i in candidate_issue_indices[:30]]

        raw_payload = {
            "dataset_path": str(dataset_path),
            "reference_path": str(reference_path),
            "target": target,
            "feature_mode": feature_mode,
            "feature_columns": feature_columns,
            "n_used": len(candidate_df),
            "n_ref_used": len(reference_df),
            "threshold_quantile": threshold_quantile,
            "issue_count": issue_count,
            "reference_issue_count": reference_issue_count,
            "issue_rate": issue_rate,
            "reference_issue_rate": reference_issue_rate,
            "excess_issue_rate": excess,
            "decision": decision,
            "risk_score": risk,
            "top_issue_indices": top_original_indices,
            "top_issue_scores": [float(value) for value in candidate_issue_scores[:30]],
        }
        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ArtResult(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_path=str(dataset_path),
            expected=expected,
            target=target,
            n_rows=n_rows,
            n_used=len(candidate_df),
            n_ref_used=len(reference_df),
            n_classes=len(set(candidate_labels.tolist())),
            feature_mode=feature_mode,
            issue_count=issue_count,
            issue_rate=issue_rate,
            clean_reference_issue_rate=reference_issue_rate,
            excess_issue_rate=excess,
            threshold_quantile=threshold_quantile,
            decision=decision,
            risk_score=round(risk, 4),
            duration_sec=round(time.perf_counter() - started, 3),
            top_issue_indices=",".join(top_original_indices),
            raw_report_path=str(raw_path),
            error=None,
        )
    except Exception as exc:
        raw_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(dataset_path),
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return ArtResult(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_path=str(dataset_path or ""),
            expected=expected,
            target=None,
            n_rows=0,
            n_used=0,
            n_ref_used=0,
            n_classes=0,
            feature_mode="",
            issue_count=None,
            issue_rate=None,
            clean_reference_issue_rate=None,
            excess_issue_rate=None,
            threshold_quantile=threshold_quantile,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=round(time.perf_counter() - started, 3),
            top_issue_indices="",
            raw_report_path=str(raw_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def metrics_for_profile(rows: list[dict[str, Any]], profile: str, decision_key: str, duration_key: str) -> dict[str, Any]:
    total = len(rows)
    expected_positive = [row for row in rows if row["expected"] == "poisoned"]
    expected_negative = [row for row in rows if row["expected"] == "clean"]
    decisions = [normalize_decision(row.get(decision_key)) for row in rows]
    counts = Counter(decisions)

    tp = sum(1 for row in rows if row["expected"] == "poisoned" and normalize_decision(row.get(decision_key)) != "ALLOW")
    fp = sum(1 for row in rows if row["expected"] == "clean" and normalize_decision(row.get(decision_key)) != "ALLOW")
    tn = sum(1 for row in rows if row["expected"] == "clean" and normalize_decision(row.get(decision_key)) == "ALLOW")
    fn = sum(1 for row in rows if row["expected"] == "poisoned" and normalize_decision(row.get(decision_key)) == "ALLOW")
    block_tp = sum(1 for row in rows if row["expected"] == "poisoned" and normalize_decision(row.get(decision_key)) == "BLOCK")
    block_fp = sum(1 for row in rows if row["expected"] == "clean" and normalize_decision(row.get(decision_key)) == "BLOCK")
    durations = [float(row.get(duration_key) or 0.0) for row in rows]

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "profile": profile,
        "n": total,
        "poisoned_n": len(expected_positive),
        "clean_n": len(expected_negative),
        "allow": counts.get("ALLOW", 0),
        "review": counts.get("REVIEW", 0),
        "block": counts.get("BLOCK", 0),
        "missing": counts.get("MISSING", 0),
        "alert_tp": tp,
        "alert_fp": fp,
        "alert_tn": tn,
        "alert_fn": fn,
        "alert_precision": precision,
        "alert_recall": recall,
        "alert_accuracy": div(tp + tn, total),
        "alert_f1": div(2 * precision * recall, precision + recall),
        "block_precision": div(block_tp, block_tp + block_fp),
        "block_recall": div(block_tp, len(expected_positive)),
        "allow_clean_recall": div(tn, len(expected_negative)),
        "total_sec": round(sum(durations), 3),
        "avg_sec": round(statistics.mean(durations), 3) if durations else 0.0,
        "median_sec": round(statistics.median(durations), 3) if durations else 0.0,
        "p95_sec": round(p95(durations), 3),
        "min_sec": round(min(durations), 3) if durations else 0.0,
        "max_sec": round(max(durations), 3) if durations else 0.0,
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def metric_row(metric: dict[str, Any]) -> list[str]:
    decisions = f"A={metric['allow']} / R={metric['review']} / B={metric['block']} / M={metric['missing']}"
    return [
        str(metric["profile"]),
        decisions,
        pct(metric["alert_precision"]),
        pct(metric["alert_recall"]),
        pct(metric["alert_accuracy"]),
        pct(metric["block_precision"]),
        pct(metric["block_recall"]),
        pct(metric["allow_clean_recall"]),
        f"{metric['avg_sec']:.3f}s",
    ]


def latency_row(metric: dict[str, Any]) -> list[str]:
    return [
        str(metric["profile"]),
        f"{metric['total_sec']:.3f}s",
        f"{metric['avg_sec']:.3f}s",
        f"{metric['median_sec']:.3f}s",
        f"{metric['p95_sec']:.3f}s",
        f"{metric['min_sec']:.3f}s",
        f"{metric['max_sec']:.3f}s",
    ]


def build_report(
    output: Path,
    baseline_index: Path,
    baseline_mode: str,
    comparison_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    threshold_quantile: float,
    review_excess: float,
    block_excess: float,
    max_rows: int,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    errors = [row for row in comparison_rows if row.get("art_error")]
    triggered = [row for row in comparison_rows if normalize_decision(row.get("art_decision")) != "ALLOW"]

    lines = [
        "# ART Spectral Signatures benchmark",
        "",
        f"Сформировано: `{timestamp}`",
        f"Baseline нашего сканера: `{baseline_mode}` из `{baseline_index}`",
        f"Датасетов: `{len(comparison_rows)}`",
        "",
        "## Что именно измерено",
        "",
        (
            "ART прогонялся как poisoning detector на основе `SpectralSignatureDefense.spectral_signature_scores()`. "
            "Для каждого датасета я строил feature representation, считал spectral-outliers по классам и сравнивал "
            "долю outliers с чистой `dataset_4` того же семейства."
        ),
        "",
        (
            f"Порог outlier берется по clean reference quantile `{threshold_quantile}`. "
            f"`REVIEW`, если excess issue-rate >= `{review_excess}`; `BLOCK`, если >= `{block_excess}`. "
            f"Максимум строк на датасет: `{max_rows}`."
        ),
        "",
        "## Сводка",
        "",
        markdown_table(
            [
                "Profile",
                "Decisions",
                "Alert precision",
                "Alert recall",
                "Alert accuracy",
                "BLOCK precision",
                "BLOCK recall",
                "ALLOW clean recall",
                "Avg latency",
            ],
            [metric_row(metric) for metric in metrics],
        ),
        "",
        "## Время работы",
        "",
        markdown_table(
            ["Profile", "Total", "Avg", "Median", "P95", "Min", "Max"],
            [latency_row(metric) for metric in metrics],
        ),
        "",
        "## ART по датасетам",
        "",
        markdown_table(
            ["Dataset", "Expected", "Decision", "Issue rate", "Clean ref", "Excess", "Top rows", "Error"],
            [
                [
                    row["dataset_id"],
                    row["expected"],
                    row["art_decision"],
                    "" if row.get("art_issue_rate") in (None, "") else f"{float(row['art_issue_rate']):.4f}",
                    "" if row.get("art_clean_reference_issue_rate") in (None, "") else f"{float(row['art_clean_reference_issue_rate']):.4f}",
                    "" if row.get("art_excess_issue_rate") in (None, "") else f"{float(row['art_excess_issue_rate']):.4f}",
                    str(row.get("art_top_issue_indices") or "")[:120],
                    str(row.get("art_error") or ""),
                ]
                for row in comparison_rows
            ],
        ),
    ]

    if triggered:
        lines.extend(
            [
                "",
                "## Где ART дал сигнал",
                "",
                markdown_table(
                    ["Dataset", "Expected", "Decision", "Excess", "Top rows"],
                    [
                        [
                            row["dataset_id"],
                            row["expected"],
                            row["art_decision"],
                            "" if row.get("art_excess_issue_rate") in (None, "") else f"{float(row['art_excess_issue_rate']):.4f}",
                            str(row.get("art_top_issue_indices") or "")[:120],
                        ]
                        for row in triggered
                    ],
                ),
            ]
        )

    if errors:
        lines.extend(
            [
                "",
                "## Ошибки",
                "",
                markdown_table(
                    ["Dataset", "Error"],
                    [[row["dataset_id"], row["art_error"]] for row in errors],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Файлы",
            "",
            "- `art_results.csv`: raw ART spectral issue rates and calibrated decisions.",
            "- `comparison.csv`: baseline, ART и combined решения по каждому датасету.",
            "- `metrics_by_profile.csv`: метрики качества и latency.",
            "- `latency_by_dataset.csv`: время по каждому датасету.",
            "- `raw_art/`: JSON-детали ART по датасетам.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output / "art_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ART spectral signatures against all_m2 scan results.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--baseline-index", type=Path, default=None)
    parser.add_argument("--baseline-mode", default="all_m2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all rows")
    parser.add_argument("--clean-number", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--max-features", type=int, default=768)
    parser.add_argument("--svd-dims", type=int, default=96)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--review-excess", type=float, default=0.01)
    parser.add_argument("--block-excess", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    root = args.project_root.resolve()
    baseline_index = args.baseline_index.resolve() if args.baseline_index else latest_profile_index(root, args.baseline_mode)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / "outputs" / "art_benchmark" / f"{stamp}_art_spectral_vs_{args.baseline_mode}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw_art"
    raw_root.mkdir(parents=True, exist_ok=True)

    art_home = output / ".art_home"
    art_home.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(art_home)

    rows = json.loads(baseline_index.read_text(encoding="utf-8"))
    baseline_rows = [row for row in rows if row.get("mode") == args.baseline_mode]
    baseline_rows.sort(key=lambda row: (str(row.get("dataset_name")), int(row.get("dataset_number") or 0)))
    if args.limit and args.limit > 0:
        baseline_rows = baseline_rows[: args.limit]

    print(f"Project root: {root}")
    print(f"Baseline index: {baseline_index}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Output: {output}")
    print(f"Jobs: {len(baseline_rows)}")
    print("ART method: SpectralSignatureDefense.spectral_signature_scores")

    art_results: list[ArtResult] = []
    comparison_rows: list[dict[str, Any]] = []
    for index, baseline in enumerate(baseline_rows, start=1):
        dataset_id = f"{baseline['dataset_name']}/dataset_{baseline['dataset_number']}"
        print(f"[{index}/{len(baseline_rows)}] {dataset_id}", flush=True)
        art_row = run_art_one(
            root=root,
            index_row=baseline,
            raw_root=raw_root,
            clean_number=args.clean_number,
            max_rows=args.max_rows,
            max_features=args.max_features,
            svd_dims=args.svd_dims,
            threshold_quantile=args.threshold_quantile,
            review_excess=args.review_excess,
            block_excess=args.block_excess,
            random_state=args.random_state,
        )
        art_results.append(art_row)

        baseline_decision = normalize_decision(baseline.get("decision"))
        art_decision = normalize_decision(art_row.decision)
        combined_decision = stronger_decision(baseline_decision, art_decision)
        baseline_duration = float(baseline.get("duration_sec") or 0.0)
        comparison_rows.append(
            {
                "dataset_name": baseline["dataset_name"],
                "dataset_number": int(baseline["dataset_number"]),
                "dataset_id": dataset_id,
                "expected": baseline.get("expected") or art_row.expected,
                "baseline_profile": args.baseline_mode,
                "baseline_decision": baseline_decision,
                "baseline_risk_score": baseline.get("risk_score"),
                "baseline_duration_sec": round(baseline_duration, 3),
                "art_decision": art_decision,
                "art_risk_score": art_row.risk_score,
                "art_duration_sec": art_row.duration_sec,
                "combined_decision": combined_decision,
                "combined_duration_sec": round(baseline_duration + art_row.duration_sec, 3),
                "art_target": art_row.target,
                "art_feature_mode": art_row.feature_mode,
                "art_n_used": art_row.n_used,
                "art_issue_count": art_row.issue_count,
                "art_issue_rate": art_row.issue_rate,
                "art_clean_reference_issue_rate": art_row.clean_reference_issue_rate,
                "art_excess_issue_rate": art_row.excess_issue_rate,
                "art_top_issue_indices": art_row.top_issue_indices,
                "art_raw_report_path": art_row.raw_report_path,
                "art_error": art_row.error,
            }
        )
        excess = "" if art_row.excess_issue_rate is None else f"{art_row.excess_issue_rate:.4f}"
        print(
            "  "
            f"art={art_decision} risk={float(art_row.risk_score or 0):.2f} excess={excess} "
            f"time={art_row.duration_sec:.3f}s {args.baseline_mode}={baseline_decision} combined={combined_decision}",
            flush=True,
        )

    art_dicts = [asdict(row) for row in art_results]
    write_csv(output / "art_results.csv", art_dicts)
    write_csv(output / "comparison.csv", comparison_rows)

    metrics = [
        metrics_for_profile(comparison_rows, "art_spectral", "art_decision", "art_duration_sec"),
        metrics_for_profile(comparison_rows, args.baseline_mode, "baseline_decision", "baseline_duration_sec"),
        metrics_for_profile(comparison_rows, f"{args.baseline_mode}+art_spectral", "combined_decision", "combined_duration_sec"),
    ]
    write_csv(output / "metrics_by_profile.csv", metrics)
    write_csv(output / "metrics_by_profile_with_latency.csv", metrics)
    latency_rows = [
        {
            "dataset_id": row["dataset_id"],
            "expected": row["expected"],
            "art_sec": row["art_duration_sec"],
            "baseline_sec": row["baseline_duration_sec"],
            "combined_sec": row["combined_duration_sec"],
        }
        for row in comparison_rows
    ]
    write_csv(output / "latency_by_dataset.csv", latency_rows)
    (output / "latency_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(
        output,
        baseline_index,
        args.baseline_mode,
        comparison_rows,
        metrics,
        threshold_quantile=args.threshold_quantile,
        review_excess=args.review_excess,
        block_excess=args.block_excess,
        max_rows=args.max_rows,
    )

    print("DONE")
    print(f"Report: {output / 'art_report.md'}")
    print(f"Comparison: {output / 'comparison.csv'}")
    print(f"Metrics: {output / 'metrics_by_profile.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)

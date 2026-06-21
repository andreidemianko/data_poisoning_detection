from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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
class CleanlabResult:
    dataset_name: str
    dataset_number: int
    dataset_path: str
    expected: str
    target: str | None
    n_rows: int
    n_used: int
    n_classes: int
    feature_mode: str
    issue_count: int | None
    issue_rate: float | None
    clean_reference_issue_rate: float | None
    excess_issue_rate: float | None
    decision: str | None
    risk_score: float | None
    duration_sec: float
    top_issue_indices: str
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
        return df
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


def build_estimator(df: pd.DataFrame, feature_mode: str, feature_columns: list[str]):
    clf = LogisticRegression(max_iter=800, class_weight="balanced")
    if feature_mode == "text":
        return Pipeline(
            [
                ("text", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
                ("clf", clf),
            ]
        )

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])


def make_features(df: pd.DataFrame, feature_mode: str, feature_columns: list[str]):
    if feature_mode == "text":
        return df[feature_columns].fillna("").astype(str).agg(" ".join, axis=1).to_numpy()
    return df[feature_columns]


def decision_from_excess(excess: float, review_excess: float, block_excess: float) -> tuple[str, float]:
    risk = min(1.0, div(excess, block_excess))
    if excess >= block_excess:
        return "BLOCK", risk
    if excess >= review_excess:
        return "REVIEW", risk
    return "ALLOW", risk


def run_cleanlab_one(
    path: Path,
    clean_number: int,
    max_rows: int,
    cv_folds: int,
    review_excess: float,
    block_excess: float,
    random_state: int,
) -> CleanlabResult:
    started = time.perf_counter()
    dataset_name = path.parent.name
    number = scan_runner.dataset_number(path)
    expected = expected_by_number(number, clean_number)
    df = load_dataset(path)
    n_rows = len(df)
    target = infer_target(df, dataset_name)
    drop = list(scan_runner.TABULAR_CONFIGS.get(dataset_name, {}).get("drop", []))

    try:
        if target is None or target not in df.columns:
            raise ValueError("missing target column")
        df = df.dropna(subset=[target]).copy()
        df = stratified_sample(df, target, max_rows, random_state)
        labels_raw = df[target].astype(str)
        counts = labels_raw.value_counts()
        if len(counts) < 2:
            raise ValueError("need at least two classes")
        cv = min(cv_folds, int(counts.min()))
        if cv < 2:
            raise ValueError("need at least two examples in every class")

        feature_mode, feature_columns = choose_feature_columns(df, target, drop, dataset_name)
        if not feature_columns:
            raise ValueError("no usable feature columns")
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels_raw)
        x = make_features(df, feature_mode, feature_columns)
        estimator = build_estimator(df, feature_mode, feature_columns)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        pred_probs = cross_val_predict(
            estimator,
            x,
            labels,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )
        issue_indices = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
            filter_by="prune_by_noise_rate",
            n_jobs=1,
        )
        issue_indices = np.asarray(issue_indices, dtype=int)
        issue_rate = div(len(issue_indices), len(df))
        return CleanlabResult(
            dataset_name=dataset_name,
            dataset_number=number,
            dataset_path=scan_runner.path_arg(path, project_root()),
            expected=expected,
            target=target,
            n_rows=n_rows,
            n_used=len(df),
            n_classes=len(counts),
            feature_mode=feature_mode,
            issue_count=int(len(issue_indices)),
            issue_rate=issue_rate,
            clean_reference_issue_rate=None,
            excess_issue_rate=None,
            decision=None,
            risk_score=None,
            duration_sec=round(time.perf_counter() - started, 3),
            top_issue_indices=",".join(str(int(df.index[i])) for i in issue_indices[:30]),
            error=None,
        )
    except Exception as exc:
        return CleanlabResult(
            dataset_name=dataset_name,
            dataset_number=number,
            dataset_path=scan_runner.path_arg(path, project_root()),
            expected=expected,
            target=target,
            n_rows=n_rows,
            n_used=0,
            n_classes=0,
            feature_mode="",
            issue_count=None,
            issue_rate=None,
            clean_reference_issue_rate=None,
            excess_issue_rate=None,
            decision="MISSING",
            risk_score=None,
            duration_sec=round(time.perf_counter() - started, 3),
            top_issue_indices="",
            error=str(exc),
        )


def calibrate_results(results: list[CleanlabResult], clean_number: int, review_excess: float, block_excess: float) -> None:
    clean_rates = {
        row.dataset_name: row.issue_rate
        for row in results
        if row.dataset_number == clean_number and row.issue_rate is not None
    }
    for row in results:
        if row.issue_rate is None:
            row.decision = "MISSING"
            continue
        reference = clean_rates.get(row.dataset_name, 0.0) or 0.0
        excess = max(0.0, row.issue_rate - reference)
        row.clean_reference_issue_rate = reference
        row.excess_issue_rate = excess
        row.decision, row.risk_score = decision_from_excess(excess, review_excess, block_excess)


def load_baseline(index_path: Path, mode: str) -> dict[tuple[str, int], dict[str, Any]]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        if row.get("mode") == mode:
            out[(str(row.get("dataset_name")), int(row.get("dataset_number")))] = row
    return out


def metric_for(rows: list[dict[str, Any]], decision_key: str, positive_expected: str, positive_decisions: set[str]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        actual = row["expected"] == positive_expected
        pred = normalize_decision(row.get(decision_key)) in positive_decisions
        if actual and pred:
            tp += 1
        elif not actual and pred:
            fp += 1
        elif not actual and not pred:
            tn += 1
        else:
            fn += 1
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    accuracy = div(tp + tn, tp + fp + tn + fn)
    f1 = div(2 * precision * recall, precision + recall)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}


def build_comparison_rows(cleanlab_results: list[CleanlabResult], baseline: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in cleanlab_results:
        base = baseline.get((result.dataset_name, result.dataset_number), {})
        baseline_decision = normalize_decision(base.get("decision")) if base else "MISSING"
        cleanlab_decision = normalize_decision(result.decision)
        combined_decision = stronger_decision(baseline_decision, cleanlab_decision)
        rows.append(
            {
                "dataset_name": result.dataset_name,
                "dataset_number": result.dataset_number,
                "expected": result.expected,
                "baseline_decision": baseline_decision,
                "cleanlab_decision": cleanlab_decision,
                "combined_decision": combined_decision,
                "cleanlab_issue_rate": result.issue_rate,
                "clean_reference_issue_rate": result.clean_reference_issue_rate,
                "cleanlab_excess_issue_rate": result.excess_issue_rate,
                "cleanlab_risk_score": result.risk_score,
                "cleanlab_error": result.error,
            }
        )
    return rows


def summarize_profile(rows: list[dict[str, Any]], decision_key: str, name: str) -> dict[str, Any]:
    decisions = Counter(normalize_decision(row.get(decision_key)) for row in rows)
    alert = metric_for(rows, decision_key, "poisoned", {"REVIEW", "BLOCK"})
    block = metric_for(rows, decision_key, "poisoned", {"BLOCK"})
    allow_clean = metric_for(rows, decision_key, "clean", {"ALLOW"})
    return {
        "profile": name,
        "n": len(rows),
        "ALLOW": decisions.get("ALLOW", 0),
        "REVIEW": decisions.get("REVIEW", 0),
        "BLOCK": decisions.get("BLOCK", 0),
        "MISSING": decisions.get("MISSING", 0),
        "alert_precision": alert["precision"],
        "alert_recall": alert["recall"],
        "alert_accuracy": alert["accuracy"],
        "alert_f1": alert["f1"],
        "block_precision": block["precision"],
        "block_recall": block["recall"],
        "allow_clean_precision": allow_clean["precision"],
        "allow_clean_recall": allow_clean["recall"],
    }


def markdown_report(output_root: Path, cleanlab_results: list[CleanlabResult], comparison: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# Cleanlab benchmark")
    lines.append("")
    lines.append(f"Сформировано: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"Датасетов: `{len(cleanlab_results)}`")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    lines.append("| Profile | Decisions | Alert precision | Alert recall | Alert accuracy | BLOCK precision | BLOCK recall | ALLOW clean recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        decisions = f"A={row['ALLOW']} / R={row['REVIEW']} / B={row['BLOCK']} / M={row['MISSING']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["profile"],
                    decisions,
                    pct(float(row["alert_precision"])),
                    pct(float(row["alert_recall"])),
                    pct(float(row["alert_accuracy"])),
                    pct(float(row["block_precision"])),
                    pct(float(row["block_recall"])),
                    pct(float(row["allow_clean_recall"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Cleanlab по датасетам")
    lines.append("")
    lines.append("| Dataset | Expected | Decision | Issue rate | Clean ref | Excess | Top issue rows | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for row in cleanlab_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row.dataset_name}/dataset_{row.dataset_number}",
                    row.expected,
                    normalize_decision(row.decision),
                    "" if row.issue_rate is None else f"{row.issue_rate:.4f}",
                    "" if row.clean_reference_issue_rate is None else f"{row.clean_reference_issue_rate:.4f}",
                    "" if row.excess_issue_rate is None else f"{row.excess_issue_rate:.4f}",
                    row.top_issue_indices[:120],
                    (row.error or "").replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Файлы")
    lines.append("")
    lines.append("- `cleanlab_results.csv`: raw cleanlab issue rates and calibrated decisions.")
    lines.append("- `comparison.csv`: baseline, cleanlab, and combined decisions per dataset.")
    lines.append("- `metrics_by_profile.csv`: metric summary.")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    first = rows[0]
    if hasattr(first, "__dataclass_fields__"):
        fieldnames = list(asdict(first).keys())
    else:
        fieldnames = list(first.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if hasattr(row, "__dataclass_fields__") else row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark cleanlab label-issue detector on project datasets.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--clean-number", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--review-excess", type=float, default=0.005)
    parser.add_argument("--block-excess", type=float, default=0.02)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--baseline-index", default=None)
    parser.add_argument("--baseline-mode", default="all_m2")
    args = parser.parse_args()

    root = project_root()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    datasets = scan_runner.list_datasets(data_root.resolve())
    selected = datasets[args.start_at - 1 :]
    if args.limit:
        selected = selected[: args.limit]

    output_root = Path(args.output) if args.output else root / "outputs" / "cleanlab_benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {root}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Datasets: {len(datasets)}")
    print(f"Selected: {len(selected)}")
    print(f"Thresholds: review_excess={args.review_excess}, block_excess={args.block_excess}")

    results: list[CleanlabResult] = []
    for index, path in enumerate(selected, start=args.start_at):
        print(f"\n[{index}/{len(datasets)}] {path.parent.name}/{path.name}")
        result = run_cleanlab_one(
            path=path,
            clean_number=args.clean_number,
            max_rows=args.max_rows,
            cv_folds=args.cv,
            review_excess=args.review_excess,
            block_excess=args.block_excess,
            random_state=args.random_state,
        )
        results.append(result)
        calibrate_results(results, args.clean_number, args.review_excess, args.block_excess)
        write_csv(output_root / "cleanlab_results.csv", results)
        print(
            f"  decision={result.decision} issue_rate={result.issue_rate} "
            f"excess={result.excess_issue_rate} time={result.duration_sec}s error={result.error}"
        )

    calibrate_results(results, args.clean_number, args.review_excess, args.block_excess)
    write_csv(output_root / "cleanlab_results.csv", results)

    baseline = {}
    if args.baseline_index:
        baseline_path = Path(args.baseline_index)
        if not baseline_path.is_absolute():
            baseline_path = root / baseline_path
        baseline = load_baseline(baseline_path, args.baseline_mode)
    comparison = build_comparison_rows(results, baseline)
    summaries = [summarize_profile(comparison, "cleanlab_decision", "cleanlab")]
    if baseline:
        summaries.append(summarize_profile(comparison, "baseline_decision", args.baseline_mode))
        summaries.append(summarize_profile(comparison, "combined_decision", f"{args.baseline_mode}+cleanlab"))

    write_csv(output_root / "comparison.csv", comparison)
    write_csv(output_root / "metrics_by_profile.csv", summaries)
    (output_root / "metrics_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "cleanlab_report.md").write_text(markdown_report(output_root, results, comparison, summaries), encoding="utf-8")

    print("\nDONE")
    print(f"Report: {output_root / 'cleanlab_report.md'}")
    print(f"Metrics: {output_root / 'metrics_by_profile.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

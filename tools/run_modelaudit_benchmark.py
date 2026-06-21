from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any


DECISION_ORDER = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2, "MISSING": -1}
SECURITY_ISSUE_SEVERITIES = {"warning", "critical"}


@dataclass
class ModelAuditRow:
    dataset_name: str
    dataset_number: int
    dataset_id: str
    expected: str
    model_path: str
    model_exists: bool
    model_size_mb: float | None
    scanner: str | None
    success: bool | None
    decision: str
    risk_score: float
    duration_sec: float
    issue_count: int
    critical_count: int
    warning_count: int
    info_count: int
    debug_count: int
    failed_checks: int | None
    top_issues: str
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


def stronger_decision(left: Any, right: Any) -> str:
    left_decision = normalize_decision(left)
    right_decision = normalize_decision(right)
    if DECISION_ORDER[left_decision] >= DECISION_ORDER[right_decision]:
        return left_decision
    return right_decision


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_model_path(root: Path, index_row: dict[str, Any]) -> Path | None:
    report_path = resolve_project_path(root, index_row.get("copied_report_path") or index_row.get("report_path"))
    if report_path and report_path.exists():
        try:
            report = load_json(report_path)
            model_path = report.get("model_path")
            if model_path:
                return resolve_project_path(root, str(model_path))
        except Exception:
            pass

    # Fallback for older indexes where model_path was stored directly.
    model_path = index_row.get("model_path")
    if model_path:
        return resolve_project_path(root, str(model_path))
    return None


def severity_value(issue: dict[str, Any]) -> str:
    severity = issue.get("severity", "")
    if isinstance(severity, dict):
        severity = severity.get("value", "")
    return str(severity).lower()


def decision_from_modelaudit(scan_dict: dict[str, Any]) -> tuple[str, float, Counter[str]]:
    issues = [issue for issue in scan_dict.get("issues", []) if isinstance(issue, dict)]
    counts = Counter(severity_value(issue) for issue in issues)
    critical = counts.get("critical", 0)
    warning = counts.get("warning", 0)
    success = bool(scan_dict.get("success", True))

    if critical:
        return "BLOCK", min(1.0, 0.9 + 0.02 * min(5, critical - 1)), counts
    if warning:
        return "REVIEW", min(0.85, 0.35 + 0.05 * min(10, warning)), counts
    if not success:
        return "REVIEW", 0.35, counts
    return "ALLOW", 0.0, counts


def compact_top_issues(issues: list[dict[str, Any]], limit: int = 5) -> str:
    visible = [issue for issue in issues if severity_value(issue) in SECURITY_ISSUE_SEVERITIES]
    if not visible:
        return ""
    parts: list[str] = []
    for issue in visible[:limit]:
        severity = severity_value(issue).upper()
        code = issue.get("rule_code") or issue.get("type") or "issue"
        message = str(issue.get("message") or "").replace("\n", " ").strip()
        if len(message) > 120:
            message = message[:117] + "..."
        parts.append(f"{severity} {code}: {message}")
    return " | ".join(parts)


def run_modelaudit_one(root: Path, row: dict[str, Any], raw_root: Path, no_whitelist: bool) -> ModelAuditRow:
    from modelaudit.core import scan_file

    dataset_name = str(row["dataset_name"])
    dataset_number = int(row["dataset_number"])
    dataset_id = f"{dataset_name}/dataset_{dataset_number}"
    expected = str(row.get("expected") or ("clean" if dataset_number == 4 else "poisoned"))
    model_path = extract_model_path(root, row)
    raw_path = raw_root / dataset_name / f"dataset_{dataset_number}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        return ModelAuditRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            model_path="",
            model_exists=False,
            model_size_mb=None,
            scanner=None,
            success=None,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=0.0,
            issue_count=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            debug_count=0,
            failed_checks=None,
            top_issues="",
            raw_report_path=str(raw_path),
            error="missing_model_path",
        )

    model_exists = model_path.exists()
    model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 3) if model_exists else None
    if not model_exists:
        return ModelAuditRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            model_path=str(model_path),
            model_exists=False,
            model_size_mb=model_size_mb,
            scanner=None,
            success=None,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=0.0,
            issue_count=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            debug_count=0,
            failed_checks=None,
            top_issues="",
            raw_report_path=str(raw_path),
            error="model_file_missing",
        )

    config: dict[str, Any] = {
        "cache_enabled": False,
        "max_file_size": 0,
        "max_total_size": 0,
    }
    if no_whitelist:
        config["no_whitelist"] = True

    started = time.perf_counter()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            scan_result = scan_file(str(model_path), config)
        duration = time.perf_counter() - started
        scan_dict = scan_result.to_dict()
        decision, risk, counts = decision_from_modelaudit(scan_dict)
        issues = [issue for issue in scan_dict.get("issues", []) if isinstance(issue, dict)]
        raw_path.write_text(json.dumps(scan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        return ModelAuditRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            model_path=str(model_path),
            model_exists=True,
            model_size_mb=model_size_mb,
            scanner=str(scan_dict.get("scanner") or ""),
            success=bool(scan_dict.get("success", True)),
            decision=decision,
            risk_score=round(risk, 4),
            duration_sec=round(duration, 3),
            issue_count=len(issues),
            critical_count=counts.get("critical", 0),
            warning_count=counts.get("warning", 0),
            info_count=counts.get("info", 0),
            debug_count=counts.get("debug", 0),
            failed_checks=scan_dict.get("failed_checks"),
            top_issues=compact_top_issues(issues),
            raw_report_path=str(raw_path),
            error=None,
        )
    except Exception as exc:
        duration = time.perf_counter() - started
        error_payload = {
            "error": type(exc).__name__,
            "message": str(exc),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        }
        raw_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ModelAuditRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            model_path=str(model_path),
            model_exists=True,
            model_size_mb=model_size_mb,
            scanner=None,
            success=False,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=round(duration, 3),
            issue_count=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            debug_count=0,
            failed_checks=None,
            top_issues="",
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
    decisions = (
        f"A={metric['allow']} / R={metric['review']} / "
        f"B={metric['block']} / M={metric['missing']}"
    )
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
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    modelaudit_rows = [row for row in comparison_rows]
    triggered = [row for row in modelaudit_rows if normalize_decision(row["modelaudit_decision"]) != "ALLOW"]
    errors = [row for row in modelaudit_rows if row.get("modelaudit_error")]

    lines: list[str] = [
        "# ModelAudit benchmark",
        "",
        f"Сформировано: `{timestamp}`",
        f"Baseline: `{baseline_mode}` из `{baseline_index}`",
        f"Датасетов: `{len(comparison_rows)}`",
        "",
        "## Что именно измерено",
        "",
        (
            "ModelAudit сканирует файл модели на security-риски: unsafe serialization, "
            "подозрительные символы, сетевой код, секреты, поврежденные или неполные артефакты. "
            "Это не label-poisoning детектор, поэтому метрики ниже считаются честно против нашей "
            "разметки `dataset_4 = clean`, но интерпретировать их надо как проверку совместимости "
            "с нашим ensemble, а не как замену cleanlab."
        ),
        "",
        "Маппинг решения: `critical -> BLOCK`, `warning -> REVIEW`, без warning/critical -> `ALLOW`.",
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
        "## Срабатывания ModelAudit",
        "",
    ]

    if triggered:
        lines.append(
            markdown_table(
                [
                    "Dataset",
                    "Expected",
                    "Decision",
                    "Risk",
                    "Scanner",
                    "Critical",
                    "Warning",
                    "Top issues",
                ],
                [
                    [
                        row["dataset_id"],
                        row["expected"],
                        row["modelaudit_decision"],
                        f"{float(row['modelaudit_risk_score']):.2f}",
                        row.get("modelaudit_scanner") or "",
                        str(row.get("modelaudit_critical_count") or 0),
                        str(row.get("modelaudit_warning_count") or 0),
                        str(row.get("modelaudit_top_issues") or ""),
                    ]
                    for row in triggered
                ],
            )
        )
    else:
        lines.append("ModelAudit не нашел warning/critical security-срабатываний на этих model artifacts.")

    if errors:
        lines.extend(
            [
                "",
                "## Ошибки",
                "",
                markdown_table(
                    ["Dataset", "Model path", "Error"],
                    [[row["dataset_id"], row["modelaudit_model_path"], row["modelaudit_error"]] for row in errors],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Файлы",
            "",
            "- `modelaudit_results.csv`: raw ModelAudit решения, severity counts и пути к raw JSON.",
            "- `comparison.csv`: baseline, ModelAudit и combined решения по каждому датасету.",
            "- `metrics_by_profile.csv`: метрики качества и latency.",
            "- `metrics_by_profile_with_latency.csv`: та же сводка в удобном виде для таблиц.",
            "- `latency_by_dataset.csv`: время по каждому датасету.",
            "- `raw_modelaudit/`: исходные JSON-результаты ModelAudit по файлам моделей.",
        ]
    )

    report = "\n".join(lines) + "\n"
    (output / "modelaudit_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark promptfoo ModelAudit against all_m2 scan results.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--baseline-index", type=Path, default=None)
    parser.add_argument("--baseline-mode", default="all_m2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all rows")
    parser.add_argument("--allow-whitelist", action="store_true", help="keep ModelAudit whitelist downgrades enabled")
    args = parser.parse_args()

    root = args.project_root.resolve()
    baseline_index = args.baseline_index.resolve() if args.baseline_index else latest_profile_index(root, args.baseline_mode)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / "outputs" / "modelaudit_benchmark" / f"{stamp}_modelaudit_vs_{args.baseline_mode}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw_modelaudit"
    raw_root.mkdir(parents=True, exist_ok=True)

    logging.getLogger("modelaudit").setLevel(logging.CRITICAL)
    rows = json.loads(baseline_index.read_text(encoding="utf-8"))
    baseline_rows = [row for row in rows if row.get("mode") == args.baseline_mode]
    baseline_rows.sort(key=lambda row: (str(row.get("dataset_name")), int(row.get("dataset_number") or 0)))
    if args.limit and args.limit > 0:
        baseline_rows = baseline_rows[: args.limit]

    print(f"Project root: {root}")
    print(f"Baseline index: {baseline_index}")
    print(f"Output: {output}")
    print(f"Jobs: {len(baseline_rows)}")

    modelaudit_results: list[ModelAuditRow] = []
    comparison_rows: list[dict[str, Any]] = []
    for index, baseline in enumerate(baseline_rows, start=1):
        dataset_id = f"{baseline['dataset_name']}/dataset_{baseline['dataset_number']}"
        print(f"[{index}/{len(baseline_rows)}] {dataset_id}", flush=True)
        modelaudit_row = run_modelaudit_one(root, baseline, raw_root, no_whitelist=not args.allow_whitelist)
        modelaudit_results.append(modelaudit_row)

        baseline_decision = normalize_decision(baseline.get("decision"))
        modelaudit_decision = normalize_decision(modelaudit_row.decision)
        combined_decision = stronger_decision(baseline_decision, modelaudit_decision)
        baseline_duration = float(baseline.get("duration_sec") or 0.0)
        comparison_rows.append(
            {
                "dataset_name": baseline["dataset_name"],
                "dataset_number": int(baseline["dataset_number"]),
                "dataset_id": dataset_id,
                "expected": baseline.get("expected") or modelaudit_row.expected,
                "baseline_profile": args.baseline_mode,
                "baseline_decision": baseline_decision,
                "baseline_risk_score": baseline.get("risk_score"),
                "baseline_duration_sec": round(baseline_duration, 3),
                "modelaudit_decision": modelaudit_decision,
                "modelaudit_risk_score": modelaudit_row.risk_score,
                "modelaudit_duration_sec": modelaudit_row.duration_sec,
                "combined_decision": combined_decision,
                "combined_duration_sec": round(baseline_duration + modelaudit_row.duration_sec, 3),
                "modelaudit_model_path": modelaudit_row.model_path,
                "modelaudit_model_size_mb": modelaudit_row.model_size_mb,
                "modelaudit_scanner": modelaudit_row.scanner,
                "modelaudit_success": modelaudit_row.success,
                "modelaudit_issue_count": modelaudit_row.issue_count,
                "modelaudit_critical_count": modelaudit_row.critical_count,
                "modelaudit_warning_count": modelaudit_row.warning_count,
                "modelaudit_info_count": modelaudit_row.info_count,
                "modelaudit_debug_count": modelaudit_row.debug_count,
                "modelaudit_failed_checks": modelaudit_row.failed_checks,
                "modelaudit_top_issues": modelaudit_row.top_issues,
                "modelaudit_raw_report_path": modelaudit_row.raw_report_path,
                "modelaudit_error": modelaudit_row.error,
            }
        )
        print(
            "  "
            f"modelaudit={modelaudit_decision} risk={modelaudit_row.risk_score:.2f} "
            f"issues={modelaudit_row.issue_count} time={modelaudit_row.duration_sec:.3f}s "
            f"combined={combined_decision}",
            flush=True,
        )

    modelaudit_dicts = [asdict(row) for row in modelaudit_results]
    write_csv(output / "modelaudit_results.csv", modelaudit_dicts)
    write_csv(output / "comparison.csv", comparison_rows)

    metrics = [
        metrics_for_profile(comparison_rows, "modelaudit", "modelaudit_decision", "modelaudit_duration_sec"),
        metrics_for_profile(comparison_rows, args.baseline_mode, "baseline_decision", "baseline_duration_sec"),
        metrics_for_profile(
            comparison_rows,
            f"{args.baseline_mode}+modelaudit",
            "combined_decision",
            "combined_duration_sec",
        ),
    ]
    write_csv(output / "metrics_by_profile.csv", metrics)
    write_csv(output / "metrics_by_profile_with_latency.csv", metrics)
    latency_rows = [
        {
            "dataset_id": row["dataset_id"],
            "expected": row["expected"],
            "modelaudit_sec": row["modelaudit_duration_sec"],
            "baseline_sec": row["baseline_duration_sec"],
            "combined_sec": row["combined_duration_sec"],
        }
        for row in comparison_rows
    ]
    write_csv(output / "latency_by_dataset.csv", latency_rows)
    (output / "latency_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(output, baseline_index, args.baseline_mode, comparison_rows, metrics)

    print("DONE")
    print(f"Report: {output / 'modelaudit_report.md'}")
    print(f"Comparison: {output / 'comparison.csv'}")
    print(f"Metrics: {output / 'metrics_by_profile.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)

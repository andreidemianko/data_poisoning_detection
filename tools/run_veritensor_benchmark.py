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
SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "WARNING", "INFO")
REVIEW_SEVERITIES = {"HIGH", "MEDIUM", "LOW", "WARNING"}
DATASET_EXTS = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"}


@dataclass
class VeritensorRow:
    dataset_name: str
    dataset_number: int
    dataset_id: str
    expected: str
    dataset_path: str
    dataset_exists: bool
    dataset_size_mb: float | None
    full_scan: bool
    decision: str
    risk_score: float
    duration_sec: float
    threat_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    warning_count: int
    info_count: int
    pii_count: int
    injection_count: int
    secret_count: int
    toxic_column_count: int
    malicious_url_count: int
    top_threats: str
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


def severity_from_threat(threat: str) -> str:
    upper = threat.upper()
    for severity in SEVERITIES:
        if upper.startswith(f"{severity}:") or f": {severity}:" in upper or f"[{severity}]" in upper:
            return severity
    return "INFO"


def threat_category_counts(threats: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for threat in threats:
        lower = threat.lower()
        if "pii leak" in lower or "pii" in lower:
            counts["pii"] += 1
        if "injection" in lower or "poisoning" in lower:
            counts["injection"] += 1
        if "secret" in lower or "password" in lower or "akia" in lower:
            counts["secret"] += 1
        if "toxic column" in lower or "article 5" in lower:
            counts["toxic_column"] += 1
        if "malicious url" in lower or "http" in lower:
            counts["malicious_url"] += 1
    return counts


def decision_from_threats(threats: list[str], scan_error: bool = False) -> tuple[str, float, Counter[str]]:
    severity_counts = Counter(severity_from_threat(threat) for threat in threats)
    if severity_counts.get("CRITICAL", 0):
        return "BLOCK", min(1.0, 0.9 + 0.02 * min(5, severity_counts["CRITICAL"] - 1)), severity_counts
    review_count = sum(severity_counts.get(severity, 0) for severity in REVIEW_SEVERITIES)
    if review_count:
        high_weight = severity_counts.get("HIGH", 0) * 0.03
        medium_weight = severity_counts.get("MEDIUM", 0) * 0.02
        low_weight = (severity_counts.get("LOW", 0) + severity_counts.get("WARNING", 0)) * 0.01
        return "REVIEW", min(0.85, 0.3 + high_weight + medium_weight + low_weight), severity_counts
    if scan_error:
        return "REVIEW", 0.35, severity_counts
    return "ALLOW", 0.0, severity_counts


def compact_top_threats(threats: list[str], limit: int = 6) -> str:
    visible = [threat for threat in threats if severity_from_threat(threat) != "INFO"]
    if not visible:
        return ""
    parts: list[str] = []
    for threat in visible[:limit]:
        clean = " ".join(str(threat).split())
        if len(clean) > 140:
            clean = clean[:137] + "..."
        parts.append(clean)
    return " | ".join(parts)


def run_veritensor_one(root: Path, row: dict[str, Any], raw_root: Path, full_scan: bool) -> VeritensorRow:
    from veritensor.engines.data.dataset_engine import scan_dataset

    dataset_name = str(row["dataset_name"])
    dataset_number = int(row["dataset_number"])
    dataset_id = f"{dataset_name}/dataset_{dataset_number}"
    expected = str(row.get("expected") or ("clean" if dataset_number == 4 else "poisoned"))
    dataset_path = resolve_project_path(root, str(row.get("dataset_path") or ""))
    raw_path = raw_root / dataset_name / f"dataset_{dataset_number}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path is None:
        return VeritensorRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            dataset_path="",
            dataset_exists=False,
            dataset_size_mb=None,
            full_scan=full_scan,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=0.0,
            threat_count=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            warning_count=0,
            info_count=0,
            pii_count=0,
            injection_count=0,
            secret_count=0,
            toxic_column_count=0,
            malicious_url_count=0,
            top_threats="",
            raw_report_path=str(raw_path),
            error="missing_dataset_path",
        )

    exists = dataset_path.exists()
    size_mb = round(dataset_path.stat().st_size / (1024 * 1024), 3) if exists else None
    if not exists:
        return VeritensorRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            dataset_path=str(dataset_path),
            dataset_exists=False,
            dataset_size_mb=size_mb,
            full_scan=full_scan,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=0.0,
            threat_count=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            warning_count=0,
            info_count=0,
            pii_count=0,
            injection_count=0,
            secret_count=0,
            toxic_column_count=0,
            malicious_url_count=0,
            top_threats="",
            raw_report_path=str(raw_path),
            error="dataset_file_missing",
        )

    if dataset_path.suffix.lower() not in DATASET_EXTS:
        raw_path.write_text(
            json.dumps({"threats": [], "error": "unsupported_dataset_extension"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return VeritensorRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            dataset_path=str(dataset_path),
            dataset_exists=True,
            dataset_size_mb=size_mb,
            full_scan=full_scan,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=0.0,
            threat_count=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            warning_count=0,
            info_count=0,
            pii_count=0,
            injection_count=0,
            secret_count=0,
            toxic_column_count=0,
            malicious_url_count=0,
            top_threats="",
            raw_report_path=str(raw_path),
            error=f"unsupported_dataset_extension:{dataset_path.suffix}",
        )

    started = time.perf_counter()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            threats, bias_data = scan_dataset(dataset_path, full_scan=full_scan)
        duration = time.perf_counter() - started
        threats = list(threats or [])
        decision, risk, severity_counts = decision_from_threats(threats)
        category_counts = threat_category_counts(threats)
        raw_payload = {
            "dataset_path": str(dataset_path),
            "full_scan": full_scan,
            "decision": decision,
            "risk_score": risk,
            "duration_sec": duration,
            "threats": threats,
            "severity_counts": dict(severity_counts),
            "category_counts": dict(category_counts),
            "bias_data": bias_data,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        }
        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return VeritensorRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            dataset_path=str(dataset_path),
            dataset_exists=True,
            dataset_size_mb=size_mb,
            full_scan=full_scan,
            decision=decision,
            risk_score=round(risk, 4),
            duration_sec=round(duration, 3),
            threat_count=len(threats),
            critical_count=severity_counts.get("CRITICAL", 0),
            high_count=severity_counts.get("HIGH", 0),
            medium_count=severity_counts.get("MEDIUM", 0),
            low_count=severity_counts.get("LOW", 0),
            warning_count=severity_counts.get("WARNING", 0),
            info_count=severity_counts.get("INFO", 0),
            pii_count=category_counts.get("pii", 0),
            injection_count=category_counts.get("injection", 0),
            secret_count=category_counts.get("secret", 0),
            toxic_column_count=category_counts.get("toxic_column", 0),
            malicious_url_count=category_counts.get("malicious_url", 0),
            top_threats=compact_top_threats(threats),
            raw_report_path=str(raw_path),
            error=None,
        )
    except Exception as exc:
        duration = time.perf_counter() - started
        raw_payload = {
            "dataset_path": str(dataset_path),
            "full_scan": full_scan,
            "error": type(exc).__name__,
            "message": str(exc),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
        }
        raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return VeritensorRow(
            dataset_name=dataset_name,
            dataset_number=dataset_number,
            dataset_id=dataset_id,
            expected=expected,
            dataset_path=str(dataset_path),
            dataset_exists=True,
            dataset_size_mb=size_mb,
            full_scan=full_scan,
            decision="REVIEW",
            risk_score=0.35,
            duration_sec=round(duration, 3),
            threat_count=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            warning_count=0,
            info_count=0,
            pii_count=0,
            injection_count=0,
            secret_count=0,
            toxic_column_count=0,
            malicious_url_count=0,
            top_threats="",
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
    full_scan: bool,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    triggered = [row for row in comparison_rows if normalize_decision(row["veritensor_decision"]) != "ALLOW"]
    errors = [row for row in comparison_rows if row.get("veritensor_error")]
    severity_totals = Counter()
    category_totals = Counter()
    for row in comparison_rows:
        severity_totals["CRITICAL"] += int(row.get("veritensor_critical_count") or 0)
        severity_totals["HIGH"] += int(row.get("veritensor_high_count") or 0)
        severity_totals["MEDIUM"] += int(row.get("veritensor_medium_count") or 0)
        severity_totals["LOW"] += int(row.get("veritensor_low_count") or 0)
        severity_totals["WARNING"] += int(row.get("veritensor_warning_count") or 0)
        severity_totals["INFO"] += int(row.get("veritensor_info_count") or 0)
        category_totals["PII"] += int(row.get("veritensor_pii_count") or 0)
        category_totals["Injection"] += int(row.get("veritensor_injection_count") or 0)
        category_totals["Secret"] += int(row.get("veritensor_secret_count") or 0)
        category_totals["Toxic column"] += int(row.get("veritensor_toxic_column_count") or 0)
        category_totals["Malicious URL"] += int(row.get("veritensor_malicious_url_count") or 0)

    if baseline_mode.startswith("all"):
        baseline_description = (
            f"Наш сканер в сравнении взят из профиля `{baseline_mode}`: это готовый прогон "
            "`--mode all`, где внутри работают несколько слоев защиты."
        )
    elif baseline_mode == "sanity":
        baseline_description = (
            "Наш сканер в сравнении взят только на первом слое `sanity`, без stats/model/all."
        )
    else:
        baseline_description = f"Наш сканер в сравнении взят из профиля `{baseline_mode}`."

    lines: list[str] = [
        "# Veritensor benchmark",
        "",
        f"Сформировано: `{timestamp}`",
        f"Baseline нашего сканера: `{baseline_mode}` из `{baseline_index}`",
        f"Датасетов: `{len(comparison_rows)}`",
        f"Режим Veritensor full_scan: `{full_scan}`",
        "",
        "## Что именно измерено",
        "",
        (
            "Veritensor прогонялся только по dataset-файлам (`csv`, `jsonl`, `parquet`). "
            f"{baseline_description}"
        ),
        "",
        (
            "Маппинг решения: `CRITICAL -> BLOCK`, `HIGH/MEDIUM/LOW/WARNING -> REVIEW`, "
            "`INFO` или отсутствие угроз -> `ALLOW`. Такой режим близок к дефолтной политике Veritensor: "
            "CRITICAL блокирует, остальные находки требуют ручной проверки."
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
        "## Что чаще всего не понравилось Veritensor",
        "",
        markdown_table(
            ["Тип", "Количество"],
            [[key, str(value)] for key, value in category_totals.most_common() if value],
        )
        if any(category_totals.values())
        else "Нет категориальных срабатываний.",
        "",
        "## Severity",
        "",
        markdown_table(
            ["Severity", "Количество"],
            [[key, str(severity_totals[key])] for key in SEVERITIES if severity_totals[key]],
        )
        if any(severity_totals.values())
        else "Нет severity-срабатываний.",
        "",
        "## Срабатывания Veritensor",
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
                    "Threats",
                    "PII",
                    "Injection",
                    "Top threats",
                ],
                [
                    [
                        row["dataset_id"],
                        row["expected"],
                        row["veritensor_decision"],
                        f"{float(row['veritensor_risk_score']):.2f}",
                        str(row.get("veritensor_threat_count") or 0),
                        str(row.get("veritensor_pii_count") or 0),
                        str(row.get("veritensor_injection_count") or 0),
                        str(row.get("veritensor_top_threats") or ""),
                    ]
                    for row in triggered
                ],
            )
        )
    else:
        lines.append("Veritensor не нашел REVIEW/BLOCK-срабатываний на этих dataset artifacts.")

    if errors:
        lines.extend(
            [
                "",
                "## Ошибки",
                "",
                markdown_table(
                    ["Dataset", "Dataset path", "Error"],
                    [[row["dataset_id"], row["veritensor_dataset_path"], row["veritensor_error"]] for row in errors],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Файлы",
            "",
            "- `veritensor_results.csv`: raw Veritensor решения, severity/category counts и пути к raw JSON.",
            f"- `comparison.csv`: профиль `{baseline_mode}`, Veritensor и combined решения по каждому датасету.",
            "- `metrics_by_profile.csv`: метрики качества и latency.",
            "- `metrics_by_profile_with_latency.csv`: та же сводка в удобном виде для таблиц.",
            "- `latency_by_dataset.csv`: время по каждому датасету.",
            "- `raw_veritensor/`: исходные JSON-результаты Veritensor по dataset-файлам.",
        ]
    )

    report = "\n".join(lines) + "\n"
    (output / "veritensor_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Veritensor against first-layer sanity scan results.")
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--baseline-index", type=Path, default=None)
    parser.add_argument("--baseline-mode", default="sanity")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all rows")
    parser.add_argument("--full-scan", action="store_true", help="scan entire datasets instead of Veritensor default 10k rows")
    args = parser.parse_args()

    root = args.project_root.resolve()
    baseline_index = args.baseline_index.resolve() if args.baseline_index else latest_profile_index(root, args.baseline_mode)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / "outputs" / "veritensor_benchmark" / f"{stamp}_veritensor_vs_{args.baseline_mode}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw_veritensor"
    raw_root.mkdir(parents=True, exist_ok=True)

    logging.getLogger("veritensor").setLevel(logging.CRITICAL)
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
    print(f"Veritensor full_scan: {args.full_scan}")

    veritensor_results: list[VeritensorRow] = []
    comparison_rows: list[dict[str, Any]] = []
    for index, baseline in enumerate(baseline_rows, start=1):
        dataset_id = f"{baseline['dataset_name']}/dataset_{baseline['dataset_number']}"
        print(f"[{index}/{len(baseline_rows)}] {dataset_id}", flush=True)
        veritensor_row = run_veritensor_one(root, baseline, raw_root, full_scan=args.full_scan)
        veritensor_results.append(veritensor_row)

        baseline_decision = normalize_decision(baseline.get("decision"))
        veritensor_decision = normalize_decision(veritensor_row.decision)
        combined_decision = stronger_decision(baseline_decision, veritensor_decision)
        baseline_duration = float(baseline.get("duration_sec") or 0.0)
        comparison_rows.append(
            {
                "dataset_name": baseline["dataset_name"],
                "dataset_number": int(baseline["dataset_number"]),
                "dataset_id": dataset_id,
                "expected": baseline.get("expected") or veritensor_row.expected,
                "baseline_profile": args.baseline_mode,
                "baseline_decision": baseline_decision,
                "baseline_risk_score": baseline.get("risk_score"),
                "baseline_duration_sec": round(baseline_duration, 3),
                "veritensor_decision": veritensor_decision,
                "veritensor_risk_score": veritensor_row.risk_score,
                "veritensor_duration_sec": veritensor_row.duration_sec,
                "combined_decision": combined_decision,
                "combined_duration_sec": round(baseline_duration + veritensor_row.duration_sec, 3),
                "veritensor_dataset_path": veritensor_row.dataset_path,
                "veritensor_dataset_size_mb": veritensor_row.dataset_size_mb,
                "veritensor_threat_count": veritensor_row.threat_count,
                "veritensor_critical_count": veritensor_row.critical_count,
                "veritensor_high_count": veritensor_row.high_count,
                "veritensor_medium_count": veritensor_row.medium_count,
                "veritensor_low_count": veritensor_row.low_count,
                "veritensor_warning_count": veritensor_row.warning_count,
                "veritensor_info_count": veritensor_row.info_count,
                "veritensor_pii_count": veritensor_row.pii_count,
                "veritensor_injection_count": veritensor_row.injection_count,
                "veritensor_secret_count": veritensor_row.secret_count,
                "veritensor_toxic_column_count": veritensor_row.toxic_column_count,
                "veritensor_malicious_url_count": veritensor_row.malicious_url_count,
                "veritensor_top_threats": veritensor_row.top_threats,
                "veritensor_raw_report_path": veritensor_row.raw_report_path,
                "veritensor_error": veritensor_row.error,
            }
        )
        print(
            "  "
            f"veritensor={veritensor_decision} risk={veritensor_row.risk_score:.2f} "
            f"threats={veritensor_row.threat_count} time={veritensor_row.duration_sec:.3f}s "
            f"{args.baseline_mode}={baseline_decision} combined={combined_decision}",
            flush=True,
        )

    veritensor_dicts = [asdict(row) for row in veritensor_results]
    write_csv(output / "veritensor_results.csv", veritensor_dicts)
    write_csv(output / "comparison.csv", comparison_rows)

    metrics = [
        metrics_for_profile(comparison_rows, "veritensor", "veritensor_decision", "veritensor_duration_sec"),
        metrics_for_profile(comparison_rows, args.baseline_mode, "baseline_decision", "baseline_duration_sec"),
        metrics_for_profile(
            comparison_rows,
            f"{args.baseline_mode}+veritensor",
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
            "veritensor_sec": row["veritensor_duration_sec"],
            "baseline_sec": row["baseline_duration_sec"],
            "combined_sec": row["combined_duration_sec"],
        }
        for row in comparison_rows
    ]
    write_csv(output / "latency_by_dataset.csv", latency_rows)
    (output / "latency_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(output, baseline_index, args.baseline_mode, comparison_rows, metrics, full_scan=args.full_scan)

    print("DONE")
    print(f"Report: {output / 'veritensor_report.md'}")
    print(f"Comparison: {output / 'comparison.csv'}")
    print(f"Metrics: {output / 'metrics_by_profile.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)

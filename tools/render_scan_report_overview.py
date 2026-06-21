from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LABEL_CANDIDATES = (
    "label",
    "labels",
    "target",
    "y",
    "class",
    "Class",
    "Risk",
    "Exited",
    "SeriousDlqin2yrs",
    "default.payment.next.month",
    "output",
    "sentiment",
    "category",
    "intent",
)

BAD_STATUSES = {"failed", "hand_check", "review", "block", "error"}


@dataclass
class ReportSummary:
    report_file: str
    run_id: str | None
    dataset_path: str | None
    model_path: str | None
    overall_status: str | None
    decision: str | None
    risk_score: float | None
    reasons: str
    scanner_count: int
    disliked_scanner_count: int


@dataclass
class ScannerRow:
    report_file: str
    run_id: str | None
    dataset_path: str | None
    model_path: str | None
    scanner: str
    layer: str
    status: str
    passed: bool | None
    decision: str | None
    risk_score: float | None
    reason: str | None
    finding_counts: str


@dataclass
class FindingRow:
    report_file: str
    run_id: str | None
    dataset_path: str | None
    model_path: str | None
    layer: str
    scanner: str
    status: str
    decision: str | None
    finding_category: str
    subtype: str | None
    rule_id: str | None
    detector: str | None
    severity: str | None
    message: str
    row_index: str | None
    label: str | None
    column: str | None
    confidence: str | None
    flagged: str | None
    flagged_fraction: str | None
    evidence: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def truncate(value: Any, limit: int = 220) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def norm_status(value: Any) -> str:
    return str(value or "").strip().lower()


def is_disliked_status(status: str) -> bool:
    return norm_status(status) in {"failed", "hand_check", "review", "block", "error"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_reports(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    patterns = ("scan_report.json", "scan_report_*.json", "*.scan_report.json")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(input_path.rglob(pattern))
    dedup = sorted({path.resolve() for path in found})
    return [Path(path) for path in dedup]


def resolve_dataset_path(raw: str | None, report_path: Path) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    roots = [project_root(), report_path.parent, Path.cwd()]
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in roots:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    return None


def read_dataset_labels(dataset_path: Path | None) -> tuple[dict[int, str], str | None]:
    if dataset_path is None or not dataset_path.exists():
        return {}, None
    try:
        import pandas as pd

        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".jsonl":
            df = pd.read_json(dataset_path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(dataset_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            return {}, None
    except Exception:
        return {}, None

    label_col = find_label_column_from_columns(list(df.columns))
    if label_col is None:
        return {}, None
    labels = {int(i): str(v) for i, v in df[label_col].items()}
    return labels, label_col


def find_label_column_from_columns(columns: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def label_from_evidence_row(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    label_col = find_label_column_from_columns([str(key) for key in row.keys()])
    if label_col and label_col in row:
        return str(row[label_col])
    return None


def compact_json(value: Any, limit: int = 260) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return truncate(text, limit)


def extract_final(report: dict[str, Any]) -> tuple[str | None, float | None, list[str]]:
    final = (report.get("metadata") or {}).get("final_decision") or {}
    if isinstance(final, dict) and final:
        return (
            str(final.get("decision")).upper() if final.get("decision") else None,
            as_float(final.get("risk_score")),
            [str(item) for item in final.get("reasons", [])],
        )
    status = report.get("overall_status")
    if status == "passed":
        return "ALLOW", None, []
    if status == "hand_check":
        return "REVIEW", None, []
    if status == "failed":
        return "BLOCK", None, []
    return None, None, []


def parse_report(report_path: Path) -> tuple[ReportSummary, list[ScannerRow], list[FindingRow]]:
    report = read_json(report_path)
    run_id = report.get("run_id")
    dataset_path = report.get("dataset_path")
    model_path = report.get("model_path")
    decision, risk, reasons = extract_final(report)
    results = report.get("results") or []
    resolved_dataset = resolve_dataset_path(dataset_path, report_path)
    labels_by_index, _label_col = read_dataset_labels(resolved_dataset)

    scanner_rows: list[ScannerRow] = []
    finding_rows: list[FindingRow] = []

    for result in results:
        details = result.get("details") or {}
        scanner_name = str(result.get("name") or "")
        layer = str(result.get("category") or "")
        status = str(result.get("status") or "")
        scanner_decision = details.get("decision")
        scanner_risk = as_float(details.get("risk_score"))
        finding_counts = details.get("finding_counts") or {}
        scanner_rows.append(
            ScannerRow(
                report_file=str(report_path),
                run_id=run_id,
                dataset_path=dataset_path,
                model_path=model_path,
                scanner=scanner_name,
                layer=layer,
                status=status,
                passed=result.get("passed"),
                decision=str(scanner_decision).upper() if scanner_decision else None,
                risk_score=scanner_risk,
                reason=details.get("reason"),
                finding_counts=compact_json(finding_counts, 500),
            )
        )

        extract_dataset_guard_findings(
            report_path,
            run_id,
            dataset_path,
            model_path,
            layer,
            scanner_name,
            status,
            details,
            labels_by_index,
            finding_rows,
        )
        extract_model_findings(
            report_path,
            run_id,
            dataset_path,
            model_path,
            layer,
            scanner_name,
            status,
            details,
            labels_by_index,
            finding_rows,
        )
        extract_stats_findings(
            report_path,
            run_id,
            dataset_path,
            model_path,
            layer,
            scanner_name,
            status,
            details,
            finding_rows,
        )

    disliked_count = sum(1 for row in scanner_rows if is_disliked_status(row.status))
    summary = ReportSummary(
        report_file=str(report_path),
        run_id=run_id,
        dataset_path=dataset_path,
        model_path=model_path,
        overall_status=report.get("overall_status"),
        decision=decision,
        risk_score=risk,
        reasons=" | ".join(reasons[:8]),
        scanner_count=len(scanner_rows),
        disliked_scanner_count=disliked_count,
    )
    return summary, scanner_rows, finding_rows


def add_finding(finding_rows: list[FindingRow], **kwargs: Any) -> None:
    finding_rows.append(FindingRow(**kwargs))


def extract_dataset_guard_findings(
    report_path: Path,
    run_id: str | None,
    dataset_path: str | None,
    model_path: str | None,
    layer: str,
    scanner_name: str,
    status: str,
    details: dict[str, Any],
    labels_by_index: dict[int, str],
    finding_rows: list[FindingRow],
) -> None:
    dg = details.get("dataset_guard_report") or {}
    files = dg.get("files") or []
    for file_item in files:
        for finding in file_item.get("top_findings") or []:
            metadata = finding.get("metadata") or {}
            row_index = finding.get("row_index")
            label = None
            if row_index is not None:
                try:
                    label = labels_by_index.get(int(row_index))
                except Exception:
                    label = None
            if label is None:
                label = label_from_evidence_row(metadata.get("evidence_row"))
            if label is None and metadata.get("top_label") is not None:
                label = str(metadata.get("top_label"))
            evidence = metadata.get("evidence_value") or metadata.get("evidence_row") or metadata
            add_finding(
                finding_rows,
                report_file=str(report_path),
                run_id=run_id,
                dataset_path=dataset_path,
                model_path=model_path,
                layer=layer,
                scanner=scanner_name,
                status=status,
                decision=str(details.get("decision")).upper() if details.get("decision") else None,
                finding_category=str(finding.get("category") or "dataset_guard"),
                subtype=finding.get("subtype"),
                rule_id=finding.get("rule_id"),
                detector=finding.get("detector"),
                severity=finding.get("severity"),
                message=str(finding.get("message") or ""),
                row_index=str(row_index) if row_index is not None else None,
                label=label,
                column=finding.get("column"),
                confidence=str(finding.get("confidence")) if finding.get("confidence") is not None else None,
                flagged=None,
                flagged_fraction=None,
                evidence=truncate(compact_json(evidence, 500), 320),
            )


def top_rows_to_labels(rows: Any, labels_by_index: dict[int, str]) -> list[str]:
    labels: list[str] = []
    if not isinstance(rows, list):
        return labels
    for item in rows[:100]:
        index = None
        if isinstance(item, int):
            index = item
        elif isinstance(item, dict):
            for key in ("row_index", "index", "row", "id"):
                if key in item:
                    index = item[key]
                    break
        try:
            if index is not None and int(index) in labels_by_index:
                labels.append(labels_by_index[int(index)])
        except Exception:
            continue
    return labels


def extract_model_findings(
    report_path: Path,
    run_id: str | None,
    dataset_path: str | None,
    model_path: str | None,
    layer: str,
    scanner_name: str,
    status: str,
    details: dict[str, Any],
    labels_by_index: dict[int, str],
    finding_rows: list[FindingRow],
) -> None:
    findings = details.get("findings") or []
    for finding in findings:
        f_status = str(finding.get("status") or "")
        if norm_status(f_status) in {"passed", "skipped"}:
            continue
        labels = top_rows_to_labels(finding.get("top_suspicious_rows"), labels_by_index)
        label = ",".join(sorted(set(labels))) if labels else None
        add_finding(
            finding_rows,
            report_file=str(report_path),
            run_id=run_id,
            dataset_path=dataset_path,
            model_path=model_path,
            layer=layer or "model",
            scanner=scanner_name,
            status=f_status,
            decision=str(details.get("decision")).upper() if details.get("decision") else None,
            finding_category=str(finding.get("category") or "model"),
            subtype=None,
            rule_id=None,
            detector=finding.get("detector"),
            severity=f_status.upper() if f_status else None,
            message=str(finding.get("verdict") or ""),
            row_index=compact_json(finding.get("top_suspicious_rows"), 220),
            label=label,
            column=None,
            confidence=None,
            flagged=str(finding.get("flagged")) if finding.get("flagged") is not None else None,
            flagged_fraction=str(finding.get("flagged_fraction")) if finding.get("flagged_fraction") is not None else None,
            evidence=compact_json(
                {
                    "trigger_tokens": finding.get("trigger_tokens"),
                    "examples": finding.get("examples"),
                    "max_minority_silhouette": finding.get("max_minority_silhouette"),
                },
                320,
            ),
        )


def extract_stats_findings(
    report_path: Path,
    run_id: str | None,
    dataset_path: str | None,
    model_path: str | None,
    layer: str,
    scanner_name: str,
    status: str,
    details: dict[str, Any],
    finding_rows: list[FindingRow],
) -> None:
    if layer != "stats" or not is_disliked_status(status):
        return
    anomalous = details.get("anomalous_classes")
    labels = []
    if isinstance(anomalous, dict):
        labels = [str(key) for key in anomalous.keys()]
    elif details.get("compared_classes"):
        labels = [str(item) for item in details.get("compared_classes") or []]
    if not labels:
        labels = [None]
    for label in labels:
        add_finding(
            finding_rows,
            report_file=str(report_path),
            run_id=run_id,
            dataset_path=dataset_path,
            model_path=model_path,
            layer=layer,
            scanner=scanner_name,
            status=status,
            decision=None,
            finding_category="stats",
            subtype=scanner_name.replace("Stats: ", ""),
            rule_id=None,
            detector=scanner_name,
            severity=status.upper(),
            message=details.get("reason") or scanner_name,
            row_index=None,
            label=label,
            column=None,
            confidence=None,
            flagged=None,
            flagged_fraction=None,
            evidence=compact_json(details, 420),
        )


def write_csv(path: Path, rows: Iterable[Any]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def color_for(index: int) -> str:
    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#0891b2",
        "#be185d",
        "#4b5563",
    ]
    return colors[index % len(colors)]


def write_bar_svg(path: Path, title: str, counts: Counter[str] | dict[str, int], width: int | None = None) -> None:
    items = [(str(k), int(v)) for k, v in counts.items() if int(v) > 0]
    items.sort(key=lambda item: item[0])
    n = max(1, len(items))
    width = width or max(760, 150 + n * 105)
    height = 460
    max_value = max((v for _, v in items), default=1)
    left = 70
    right = 36
    top = 70
    bottom = 118
    chart_w = width - left - right
    chart_h = height - top - bottom
    slot_w = chart_w / n
    bar_w = min(64, max(28, slot_w * 0.56))

    def y_for(value: float) -> float:
        return top + chart_h - (chart_h * value / max_value if max_value else 0)

    grid_steps = min(5, max_value)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.16"/>
  </filter>
</defs>
<style>
text{font-family:Inter,Segoe UI,Arial,Helvetica,sans-serif;font-size:13px;fill:#111827}
.title{font-size:21px;font-weight:800}
.axis{stroke:#cbd5e1;stroke-width:1}
.grid{stroke:#e5e7eb;stroke-width:1}
.muted{fill:#64748b}
.value{font-weight:700;font-size:13px}
.label{font-size:12px}
</style>
""",
        f'<rect width="{width}" height="{height}" rx="14" fill="#ffffff"/>',
        f'<text class="title" x="28" y="36">{html.escape(title)}</text>',
    ]

    if items:
        for step in range(grid_steps + 1):
            value = max_value * step / grid_steps if grid_steps else 0
            y = y_for(value)
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
            lines.append(f'<text class="muted" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{int(round(value))}</text>')

        lines.append(f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>')

        for i, (label, value) in enumerate(items):
            x_center = left + slot_w * i + slot_w / 2
            x = x_center - bar_w / 2
            y = y_for(value)
            h = top + chart_h - y
            color = color_for(i)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'rx="8" fill="{color}" filter="url(#shadow)"/>'
            )
            lines.append(f'<text class="value" x="{x_center:.1f}" y="{max(56, y - 9):.1f}" text-anchor="middle">{value}</text>')
            short_label = truncate(label, 20)
            label_y = top + chart_h + 22
            if len(short_label) > 11:
                lines.append(
                    f'<text class="label" x="{x_center:.1f}" y="{label_y:.1f}" text-anchor="end" '
                    f'transform="rotate(-35 {x_center:.1f} {label_y:.1f})">{html.escape(short_label)}</text>'
                )
            else:
                lines.append(f'<text class="label" x="{x_center:.1f}" y="{label_y:.1f}" text-anchor="middle">{html.escape(short_label)}</text>')

    if not items:
        lines.append('<text class="muted" x="28" y="80">Нет данных</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_risk_svg(path: Path, summaries: list[ReportSummary], title: str = "Распределение risk score") -> None:
    buckets = Counter()
    for summary in summaries:
        risk = summary.risk_score
        if risk is None:
            buckets["неизвестно"] += 1
        else:
            lo = math.floor(max(0.0, min(0.999, risk)) * 10) / 10
            buckets[f"{lo:.1f}-{lo + 0.1:.1f}"] += 1
    write_bar_svg(path, title, buckets)


def counter_from(rows: Iterable[Any], attr: str, default: str = "unknown") -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = getattr(row, attr, None) or default
        counter[str(value)] += 1
    return counter


def short_model_name(model_path: str | None) -> str:
    if not model_path:
        return "без модели"
    normalized = str(model_path).replace("\\", "/")
    path = Path(normalized)
    generic_files = {"model.safetensors", "model.pt", "pytorch_model.bin", "model.bin"}
    if path.name in generic_files and path.parent.name and path.parent.name != "models":
        return path.parent.name
    if path.suffix:
        return path.stem
    return path.name or normalized


def model_decision_counter(rows: list[ReportSummary]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[f"{short_model_name(row.model_path)} / {row.decision or 'unknown'}"] += 1
    return counter


def model_findings_counter(rows: list[FindingRow]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[short_model_name(row.model_path)] += 1
    return counter


def scanner_status_counter(rows: list[ScannerRow]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[f"{row.layer}:{row.status}"] += 1
    return counter


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(truncate(cell, 180).replace("|", "/") for cell in row) + " |")
    return "\n".join(out)


def model_summary_rows(summaries: list[ReportSummary], scanners: list[ScannerRow], findings: list[FindingRow]) -> list[list[Any]]:
    names = sorted(
        {short_model_name(row.model_path) for row in summaries}
        | {short_model_name(row.model_path) for row in scanners}
        | {short_model_name(row.model_path) for row in findings}
    )
    rows: list[list[Any]] = []
    for name in names:
        model_reports = [row for row in summaries if short_model_name(row.model_path) == name]
        risks = [row.risk_score for row in model_reports if row.risk_score is not None]
        decision_counts = Counter((row.decision or "unknown").upper() for row in model_reports)
        disliked_scanners = [
            row
            for row in scanners
            if short_model_name(row.model_path) == name and is_disliked_status(row.status)
        ]
        model_findings = [row for row in findings if short_model_name(row.model_path) == name]
        rows.append(
            [
                name,
                len(model_reports),
                decision_counts.get("ALLOW", 0),
                decision_counts.get("REVIEW", 0),
                decision_counts.get("BLOCK", 0),
                "" if not risks else f"{sum(risks) / len(risks):.4f}",
                len(disliked_scanners),
                len(model_findings),
            ]
        )
    return rows


def build_markdown(
    output_root: Path,
    input_path: Path,
    summaries: list[ReportSummary],
    scanners: list[ScannerRow],
    findings: list[FindingRow],
    max_findings: int,
) -> str:
    decision_counts = counter_from(summaries, "decision")
    status_counts = counter_from(summaries, "overall_status")
    category_counts = counter_from(findings, "finding_category")
    class_counts = counter_from(findings, "label")
    layer_counts = counter_from(findings, "layer")

    lines: list[str] = []
    lines.append("# Обзор отчета сканера")
    lines.append("")
    lines.append(f"Сформировано: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"Источник: `{input_path}`")
    lines.append(f"Разобрано отчетов: `{len(summaries)}`")
    lines.append(f"Результатов сканеров: `{len(scanners)}`")
    lines.append(f"Извлечено подозрительных находок: `{len(findings)}`")
    lines.append("")
    lines.append("## Графики")
    lines.append("")
    for name, title in [
        ("decisions.svg", "Финальные решения"),
        ("model_decisions.svg", "Решения по моделям"),
        ("model_findings.svg", "Находки по моделям"),
        ("scanner_status.svg", "Статусы сканеров по слоям"),
        ("findings_by_layer.svg", "Находки по слоям"),
        ("finding_categories.svg", "Категории находок"),
        ("class_distribution.svg", "Распределение находок по классам"),
        ("risk_scores.svg", "Распределение risk score"),
    ]:
        lines.append(f"![{title}](charts/{name})")
        lines.append("")

    lines.append("## Сводка финальных решений")
    lines.append("")
    lines.append(markdown_table(["Решение", "Количество"], [[k, v] for k, v in decision_counts.most_common()]))
    lines.append("")
    lines.append("## Сводка общих статусов")
    lines.append("")
    lines.append(markdown_table(["Общий статус", "Количество"], [[k, v] for k, v in status_counts.most_common()]))
    lines.append("")

    lines.append("## Сводка по моделям")
    lines.append("")
    lines.append(
        markdown_table(
            ["Модель", "Отчетов", "ALLOW", "REVIEW", "BLOCK", "Средний risk", "Проблемных сканеров", "Находок"],
            model_summary_rows(summaries, scanners, findings),
        )
    )
    lines.append("")

    lines.append("## Что не понравилось сканеру")
    lines.append("")
    lines.append("### По слоям")
    lines.append("")
    lines.append(markdown_table(["Слой", "Количество"], [[k, v] for k, v in layer_counts.most_common()]))
    lines.append("")
    lines.append("### По категориям находок")
    lines.append("")
    lines.append(markdown_table(["Категория", "Количество"], [[k, v] for k, v in category_counts.most_common()]))
    lines.append("")
    lines.append("### По классам и меткам")
    lines.append("")
    lines.append(markdown_table(["Класс/метка", "Количество"], [[k, v] for k, v in class_counts.most_common()]))
    lines.append("")

    lines.append("## Отчеты")
    lines.append("")
    report_rows = [
        [
            Path(row.report_file).name,
            row.dataset_path,
            row.model_path,
            row.overall_status,
            row.decision,
            "" if row.risk_score is None else f"{row.risk_score:.4f}",
            row.disliked_scanner_count,
            row.reasons,
        ]
        for row in summaries
    ]
    lines.append(markdown_table(["Отчет", "Датасет", "Модель", "Статус", "Решение", "Risk", "Проблемных сканеров", "Причины"], report_rows))
    lines.append("")

    disliked_scanners = [row for row in scanners if is_disliked_status(row.status)]
    lines.append("## Проблемные результаты сканеров")
    lines.append("")
    scanner_rows = [
        [
            Path(row.report_file).name,
            row.layer,
            row.scanner,
            row.status,
            row.decision,
            "" if row.risk_score is None else f"{row.risk_score:.4f}",
            row.finding_counts,
            row.reason,
        ]
        for row in disliked_scanners
    ]
    lines.append(markdown_table(["Отчет", "Слой", "Сканер", "Статус", "Решение", "Risk", "Счетчики находок", "Причина"], scanner_rows[:max_findings]))
    if len(scanner_rows) > max_findings:
        lines.append("")
        lines.append(f"Показаны первые `{max_findings}` проблемных строк сканеров. Полные данные лежат в `scanner_results.csv`.")
    lines.append("")

    lines.append("## Находки")
    lines.append("")
    finding_rows = [
        [
            Path(row.report_file).name,
            row.layer,
            row.finding_category,
            row.subtype,
            row.detector,
            row.severity or row.status,
            row.label,
            row.row_index,
            row.column,
            row.message,
            row.evidence,
        ]
        for row in findings
    ]
    lines.append(
        markdown_table(
            ["Отчет", "Слой", "Категория", "Подтип", "Детектор", "Серьезность", "Класс", "Строки", "Колонка", "Сообщение", "Доказательство"],
            finding_rows[:max_findings],
        )
    )
    if len(finding_rows) > max_findings:
        lines.append("")
        lines.append(f"Показаны первые `{max_findings}` находок. Полные данные лежат в `findings.csv`.")
    lines.append("")
    lines.append("## Файлы результата")
    lines.append("")
    lines.append("- `report_summary.csv`: одна строка на каждый разобранный отчет.")
    lines.append("- `scanner_results.csv`: одна строка на каждый результат сканера.")
    lines.append("- `findings.csv`: все извлеченные подозрительные находки и примеры.")
    lines.append("- `charts/*.svg`: графики для быстрого чтения.")
    return "\n".join(lines)


def run(input_path: Path, output_root: Path, max_findings: int) -> None:
    reports = discover_reports(input_path)
    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ReportSummary] = []
    scanners: list[ScannerRow] = []
    findings: list[FindingRow] = []

    for report_path in reports:
        summary, scanner_rows, finding_rows = parse_report(report_path)
        summaries.append(summary)
        scanners.extend(scanner_rows)
        findings.extend(finding_rows)

    write_csv(output_root / "report_summary.csv", summaries)
    write_csv(output_root / "scanner_results.csv", scanners)
    write_csv(output_root / "findings.csv", findings)

    write_bar_svg(charts_dir / "decisions.svg", "Финальные решения", counter_from(summaries, "decision"))
    write_bar_svg(charts_dir / "model_decisions.svg", "Решения по моделям", model_decision_counter(summaries))
    write_bar_svg(charts_dir / "model_findings.svg", "Находки по моделям", model_findings_counter(findings))
    write_bar_svg(charts_dir / "scanner_status.svg", "Статусы сканеров по слоям", scanner_status_counter(scanners))
    write_bar_svg(charts_dir / "findings_by_layer.svg", "Находки по слоям", counter_from(findings, "layer"))
    write_bar_svg(charts_dir / "finding_categories.svg", "Категории находок", counter_from(findings, "finding_category"))
    write_bar_svg(charts_dir / "class_distribution.svg", "Распределение находок по классам", counter_from(findings, "label"))
    write_risk_svg(charts_dir / "risk_scores.svg", summaries, "Распределение risk score")

    markdown = build_markdown(output_root, input_path, summaries, scanners, findings, max_findings)
    (output_root / "overview.md").write_text(markdown.replace("?", ""), encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "input": str(input_path),
                "reports": len(summaries),
                "scanner_rows": len(scanners),
                "findings": len(findings),
                "decision_counts": dict(counter_from(summaries, "decision")),
                "model_decision_counts": dict(model_decision_counter(summaries)),
                "model_finding_counts": dict(model_findings_counter(findings)),
                "finding_category_counts": dict(counter_from(findings, "finding_category")),
                "class_counts": dict(counter_from(findings, "label")),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать человекочитаемый обзор scan_report JSON.")
    parser.add_argument("--input", required=True, help="Файл scan_report.json или папка с отчетами.")
    parser.add_argument("--output", default=None, help="Папка вывода. По умолчанию outputs/report_overviews/<timestamp>.")
    parser.add_argument("--max-findings", type=int, default=250, help="Максимум находок в таблицах overview.md.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if args.output:
        output_root = Path(args.output).resolve()
    else:
        output_root = project_root() / "outputs" / "report_overviews" / datetime.now().strftime("%Y%m%d_%H%M%S")

    run(input_path, output_root, args.max_findings)
    print(f"Обзор: {output_root / 'overview.md'}")
    print(f"Находки CSV: {output_root / 'findings.csv'}")
    print(f"Графики: {output_root / 'charts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


LAYERS = ["sanity", "stats", "model"]
SUBJECTS = LAYERS + ["final"]
DECISIONS = ["ALLOW", "REVIEW", "BLOCK", "MISSING"]
BAD_STATUSES = {"failed", "hand_check", "review", "block", "error"}
STATUS_SEVERITY = {
    "missing": -1,
    "skipped": 0,
    "passed": 1,
    "allow": 1,
    "hand_check": 2,
    "review": 2,
    "failed": 3,
    "block": 3,
    "error": 3,
}
DECISION_COLORS = {
    "ALLOW": "#16a34a",
    "REVIEW": "#f59e0b",
    "BLOCK": "#dc2626",
    "MISSING": "#94a3b8",
}
EXPECTED_COLORS = {
    "clean": "#38bdf8",
    "poisoned": "#a855f7",
    "unknown": "#94a3b8",
}
DATASET_RE = re.compile(r"dataset_(\d+)", re.IGNORECASE)
CACHE_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)_dataset_(?P<number>\d+)_cand", re.IGNORECASE)


@dataclass
class LayerView:
    decision: str
    risk_score: float | None
    scanner_count: int
    bad_count: int
    skipped_count: int
    top_scanners: str


@dataclass
class ReportView:
    report_file: str
    dataset_name: str
    dataset_number: int | None
    dataset_id: str
    expected: str
    model_name: str
    final_decision: str
    final_risk: float | None
    overall_status: str
    duration_sec: float | None
    sanity_decision: str
    stats_decision: str
    model_decision: str
    sanity_risk: float | None
    stats_risk: float | None
    model_risk: float | None
    sanity_bad: int
    stats_bad: int
    model_bad: int
    sanity_scanners: int
    stats_scanners: int
    model_scanners: int
    sanity_top: str
    stats_top: str
    model_top: str


@dataclass
class ProblemScanner:
    report_file: str
    dataset_id: str
    expected: str
    layer: str
    scanner: str
    status: str
    reason: str


@dataclass
class TriggerRow:
    layer: str
    scanner: str
    status: str
    category: str
    subtype: str
    row_index: str
    column: str
    signal: str
    value: str
    row_data: str


def safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def normalize_decision(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"ALLOW", "REVIEW", "BLOCK"}:
        return text
    return "MISSING"


def decision_from_status(status: str | None) -> str:
    severity = STATUS_SEVERITY.get(str(status or "missing").lower(), -1)
    if severity >= 3:
        return "BLOCK"
    if severity >= 2:
        return "REVIEW"
    if severity >= 0:
        return "ALLOW"
    return "MISSING"


def severity_of(status: str | None) -> int:
    return STATUS_SEVERITY.get(str(status or "missing").lower(), -1)


def discover_reports(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    candidates: list[Path] = []
    for pattern in ("scan_report.json", "scan_report_*.json", "*.scan_report.json"):
        candidates.extend(input_path.rglob(pattern))
    seen: set[Path] = set()
    reports: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            reports.append(path)
    return sorted(reports)


def load_index(index_path: Path | None, mode: str | None) -> dict[str, dict[str, Any]]:
    if not index_path or not index_path.exists():
        return {}
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if mode and row.get("mode") != mode:
            continue
        for key in ("copied_report_path", "report_path"):
            value = row.get(key)
            if value:
                try:
                    mapping[str(Path(value).resolve()).lower()] = row
                except Exception:
                    mapping[str(value).lower()] = row
    return mapping


def index_row_for(path: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return index.get(str(path.resolve()).lower())


def infer_dataset_from_path(report_path: Path, report: dict[str, Any], index_row: dict[str, Any] | None) -> tuple[str, int | None]:
    if index_row:
        number = index_row.get("dataset_number")
        try:
            number = int(number)
        except Exception:
            number = None
        return str(index_row.get("dataset_name") or "unknown"), number

    parts = list(report_path.parts)
    for idx, part in enumerate(parts):
        match = DATASET_RE.fullmatch(part)
        if match and idx > 0:
            return parts[idx - 1], int(match.group(1))

    dataset_path = str(report.get("dataset_path") or "").replace("\\", "/")
    pieces = dataset_path.split("/")
    for idx, piece in enumerate(pieces):
        match = DATASET_RE.fullmatch(piece)
        if match and idx > 0:
            return pieces[idx - 1], int(match.group(1))

    cache_match = CACHE_RE.search(dataset_path)
    if cache_match:
        return cache_match.group("name"), int(cache_match.group("number"))

    return "unknown", None


def infer_model_name(report: dict[str, Any], index_row: dict[str, Any] | None) -> str:
    if index_row and index_row.get("model_name"):
        return str(index_row["model_name"])
    model_path = str(report.get("model_path") or "").replace("\\", "/")
    if not model_path:
        return ""
    path = Path(model_path)
    generic_files = {"model.safetensors", "model.pt", "model.bin", "pytorch_model.bin"}
    if path.name in generic_files and path.parent.name:
        return path.parent.name
    return path.stem or path.name


def expected_from_number(number: int | None, index_row: dict[str, Any] | None) -> str:
    if index_row and index_row.get("expected"):
        return str(index_row["expected"])
    if number is None:
        return "unknown"
    return "clean" if number == 4 else "poisoned"


def final_from_report(report: dict[str, Any]) -> tuple[str, float | None, dict[str, float]]:
    final = (report.get("metadata") or {}).get("final_decision") or {}
    scores: dict[str, float] = {}
    if isinstance(final, dict):
        raw_scores = final.get("category_scores") or {}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                risk = safe_float(value)
                if risk is not None:
                    scores[str(key)] = risk
        decision = normalize_decision(final.get("decision"))
        risk = safe_float(final.get("risk_score"))
        if decision != "MISSING":
            return decision, risk, scores

    status = str(report.get("overall_status") or "")
    return decision_from_status(status), None, scores


def risk_from_details(details: dict[str, Any]) -> float | None:
    risk = safe_float(details.get("risk_score"))
    if risk is not None:
        return risk
    nested = details.get("dataset_guard_report")
    if isinstance(nested, dict):
        return safe_float(nested.get("risk_score"))
    return None


def reason_from_details(details: dict[str, Any]) -> str:
    for key in ("reason", "message", "error"):
        value = details.get(key)
        if value:
            return str(value)
    return ""


def truncate_text(value: Any, limit: int = 260) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_json(value: Any, limit: int = 520) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return truncate_text(text, limit)


def resolve_dataset_path(report: dict[str, Any]) -> Path | None:
    raw = report.get("dataset_path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute() and path.exists():
        return path
    root = Path(__file__).resolve().parents[1]
    candidate = root / path
    return candidate if candidate.exists() else None


def read_csv_rows(path: Path | None, row_indices: Iterable[int], max_rows: int = 60) -> dict[int, dict[str, Any]]:
    wanted = [idx for idx in dict.fromkeys(row_indices) if isinstance(idx, int) and idx >= 0]
    wanted = wanted[:max_rows]
    if not path or path.suffix.lower() != ".csv" or not wanted:
        return {}
    wanted_set = set(wanted)
    out: dict[int, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if idx in wanted_set:
                    out[idx] = dict(row)
                    if len(out) >= len(wanted_set):
                        break
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if idx in wanted_set:
                    out[idx] = dict(row)
                    if len(out) >= len(wanted_set):
                        break
    except Exception:
        return {}
    return out


def collect_row_indices(report: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for item in report.get("results") or []:
        details = item.get("details") or {}
        nested = details.get("dataset_guard_report") or {}
        for file_item in nested.get("files") or []:
            for finding in file_item.get("top_findings") or []:
                idx = finding.get("row_index")
                if isinstance(idx, int):
                    indices.append(idx)
        for finding in details.get("findings") or []:
            for idx in finding.get("top_suspicious_rows") or []:
                if isinstance(idx, int):
                    indices.append(idx)
    return indices


def extract_trigger_rows(path: Path, limit: int = 180) -> list[TriggerRow]:
    report = json.loads(path.read_text(encoding="utf-8"))
    dataset_rows = read_csv_rows(resolve_dataset_path(report), collect_row_indices(report), max_rows=120)
    triggers: list[TriggerRow] = []

    def add(row: TriggerRow) -> None:
        if len(triggers) < limit:
            triggers.append(row)

    for item in report.get("results") or []:
        layer = str(item.get("category") or "")
        scanner = str(item.get("name") or "unnamed")
        status = str(item.get("status") or "")
        details = item.get("details") or {}

        nested = details.get("dataset_guard_report") or {}
        for file_item in nested.get("files") or []:
            for finding in file_item.get("top_findings") or []:
                meta = finding.get("metadata") or {}
                idx = finding.get("row_index")
                row_data = meta.get("evidence_row")
                if row_data is None and isinstance(idx, int):
                    row_data = dataset_rows.get(idx)
                add(
                    TriggerRow(
                        layer=layer,
                        scanner=scanner,
                        status=str(finding.get("severity") or status),
                        category=str(finding.get("category") or ""),
                        subtype=str(finding.get("subtype") or ""),
                        row_index="" if idx is None else str(idx),
                        column=str(finding.get("column") or ""),
                        signal=str(finding.get("message") or ""),
                        value=truncate_text(meta.get("evidence_value"), 220),
                        row_data=compact_json(row_data, 520),
                    )
                )

        if layer == "stats" and status.lower() in BAD_STATUSES:
            anomalous = details.get("anomalous_classes")
            if isinstance(anomalous, dict):
                for class_name, payload in anomalous.items():
                    if isinstance(payload, list):
                        for feature in payload[:10]:
                            add(
                                TriggerRow(
                                    layer=layer,
                                    scanner=scanner,
                                    status=status,
                                    category="stats",
                                    subtype=f"class {class_name}",
                                    row_index="",
                                    column=str(feature.get("feature") if isinstance(feature, dict) else ""),
                                    signal=compact_json(feature, 220),
                                    value="",
                                    row_data="",
                                )
                            )
                    elif isinstance(payload, dict):
                        examples = payload.get("examples") or payload.get("top_features") or []
                        if examples:
                            for feature in examples[:10]:
                                add(
                                    TriggerRow(
                                        layer=layer,
                                        scanner=scanner,
                                        status=status,
                                        category="stats",
                                        subtype=f"class {class_name}",
                                        row_index="",
                                        column=str(feature.get("feature") if isinstance(feature, dict) else ""),
                                        signal=compact_json(feature, 220),
                                        value="",
                                        row_data="",
                                    )
                                )
                        else:
                            add(
                                TriggerRow(
                                    layer=layer,
                                    scanner=scanner,
                                    status=status,
                                    category="stats",
                                    subtype=f"class {class_name}",
                                    row_index="",
                                    column="",
                                    signal=compact_json(payload, 260),
                                    value="",
                                    row_data="",
                                )
                            )
            elif status.lower() in BAD_STATUSES:
                add(
                    TriggerRow(
                        layer=layer,
                        scanner=scanner,
                        status=status,
                        category="stats",
                        subtype="summary",
                        row_index="",
                        column="",
                        signal=compact_json(details, 360),
                        value="",
                        row_data="",
                    )
                )

        if layer == "model":
            for finding in details.get("findings") or []:
                detector = str(finding.get("detector") or scanner)
                finding_status = str(finding.get("status") or status)
                verdict = str(finding.get("verdict") or "")
                flagged = finding.get("flagged")
                flagged_fraction = finding.get("flagged_fraction")
                signal = verdict
                if flagged is not None:
                    signal += f"; flagged={flagged}"
                if flagged_fraction is not None:
                    signal += f"; fraction={flagged_fraction}"
                rows = finding.get("top_suspicious_rows") or []
                if rows:
                    for idx in rows[:20]:
                        row_data = dataset_rows.get(idx) if isinstance(idx, int) else None
                        add(
                            TriggerRow(
                                layer=layer,
                                scanner=detector,
                                status=finding_status,
                                category=str(finding.get("category") or ""),
                                subtype="top_suspicious_row",
                                row_index=str(idx),
                                column="",
                                signal=truncate_text(signal, 240),
                                value="",
                                row_data=compact_json(row_data, 520),
                            )
                        )
                else:
                    add(
                        TriggerRow(
                            layer=layer,
                            scanner=detector,
                            status=finding_status,
                            category=str(finding.get("category") or ""),
                            subtype="detector_summary",
                            row_index="",
                            column="",
                            signal=truncate_text(signal, 260),
                            value="",
                            row_data="",
                        )
                    )
    return triggers


def aggregate_layer(results: list[dict[str, Any]], layer: str, category_scores: dict[str, float]) -> tuple[LayerView, list[ProblemScanner]]:
    layer_results = [item for item in results if item.get("category") == layer]
    if not layer_results:
        return LayerView("MISSING", category_scores.get(layer), 0, 0, 0, ""), []

    worst_status = max((str(item.get("status") or "missing") for item in layer_results), key=severity_of)
    detail_decisions = [
        normalize_decision((item.get("details") or {}).get("decision"))
        for item in layer_results
        if normalize_decision((item.get("details") or {}).get("decision")) != "MISSING"
    ]
    if detail_decisions and max(detail_decisions, key=lambda value: DECISIONS.index(value)) != "ALLOW":
        decision = max(detail_decisions, key=lambda value: DECISIONS.index(value))
    else:
        decision = decision_from_status(worst_status)

    bad = []
    skipped = 0
    risks: list[float] = []
    for item in layer_results:
        status = str(item.get("status") or "missing")
        if status.lower() == "skipped":
            skipped += 1
        details = item.get("details") or {}
        risk = risk_from_details(details)
        if risk is not None:
            risks.append(risk)
        if status.lower() in BAD_STATUSES:
            bad.append(item)

    risk_score = category_scores.get(layer)
    if risk_score is None and risks:
        risk_score = max(risks)

    top_names = [str(item.get("name") or "unnamed") for item in bad[:5]]
    return LayerView(decision, risk_score, len(layer_results), len(bad), skipped, " / ".join(top_names)), []


def parse_report(path: Path, index: dict[str, dict[str, Any]]) -> tuple[ReportView, list[ProblemScanner], set[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    index_row = index_row_for(path, index)
    dataset_name, dataset_number = infer_dataset_from_path(path, report, index_row)
    expected = expected_from_number(dataset_number, index_row)
    model_name = infer_model_name(report, index_row)
    final_decision, final_risk, category_scores = final_from_report(report)
    results = report.get("results") or []
    present_layers = {str(item.get("category")) for item in results if item.get("category")}

    layer_views: dict[str, LayerView] = {}
    problems: list[ProblemScanner] = []
    dataset_id = f"{dataset_name}/dataset_{dataset_number}" if dataset_number is not None else dataset_name
    for layer in LAYERS:
        view, _ = aggregate_layer(results, layer, category_scores)
        layer_views[layer] = view
        for item in results:
            if item.get("category") != layer:
                continue
            status = str(item.get("status") or "missing")
            if status.lower() not in BAD_STATUSES:
                continue
            problems.append(
                ProblemScanner(
                    report_file=str(path),
                    dataset_id=dataset_id,
                    expected=expected,
                    layer=layer,
                    scanner=str(item.get("name") or "unnamed"),
                    status=status,
                    reason=reason_from_details(item.get("details") or {}),
                )
            )

    duration = safe_float(index_row.get("duration_sec")) if index_row else None
    view = ReportView(
        report_file=str(path),
        dataset_name=dataset_name,
        dataset_number=dataset_number,
        dataset_id=dataset_id,
        expected=expected,
        model_name=model_name,
        final_decision=final_decision,
        final_risk=final_risk,
        overall_status=str(report.get("overall_status") or ""),
        duration_sec=duration,
        sanity_decision=layer_views["sanity"].decision,
        stats_decision=layer_views["stats"].decision,
        model_decision=layer_views["model"].decision,
        sanity_risk=layer_views["sanity"].risk_score,
        stats_risk=layer_views["stats"].risk_score,
        model_risk=layer_views["model"].risk_score,
        sanity_bad=layer_views["sanity"].bad_count,
        stats_bad=layer_views["stats"].bad_count,
        model_bad=layer_views["model"].bad_count,
        sanity_scanners=layer_views["sanity"].scanner_count,
        stats_scanners=layer_views["stats"].scanner_count,
        model_scanners=layer_views["model"].scanner_count,
        sanity_top=layer_views["sanity"].top_scanners,
        stats_top=layer_views["stats"].top_scanners,
        model_top=layer_views["model"].top_scanners,
    )
    return view, problems, present_layers


def metric_for(rows: list[ReportView], subject: str, positive_expected: str, positive_decisions: set[str]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        actual = row.expected == positive_expected
        decision = getattr(row, f"{subject}_decision") if subject != "final" else row.final_decision
        pred = decision in positive_decisions
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
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def decision_counts(rows: list[ReportView], subject: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        decision = getattr(row, f"{subject}_decision") if subject != "final" else row.final_decision
        counter[decision] += 1
    return counter


def mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def build_metrics(rows: list[ReportView]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        counts = decision_counts(rows, subject)
        alert = metric_for(rows, subject, "poisoned", {"REVIEW", "BLOCK"})
        block = metric_for(rows, subject, "poisoned", {"BLOCK"})
        allow_clean = metric_for(rows, subject, "clean", {"ALLOW"})
        risk_attr = f"{subject}_risk" if subject != "final" else "final_risk"
        out.append(
            {
                "subject": subject,
                "ALLOW": counts.get("ALLOW", 0),
                "REVIEW": counts.get("REVIEW", 0),
                "BLOCK": counts.get("BLOCK", 0),
                "MISSING": counts.get("MISSING", 0),
                "alert_precision": alert["precision"],
                "alert_recall": alert["recall"],
                "alert_accuracy": alert["accuracy"],
                "alert_f1": alert["f1"],
                "block_precision": block["precision"],
                "block_recall": block["recall"],
                "allow_clean_precision": allow_clean["precision"],
                "allow_clean_recall": allow_clean["recall"],
                "avg_risk": mean(getattr(row, risk_attr) for row in rows),
            }
        )
    return out


def write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    first = rows[0]
    fields = list(asdict(first).keys()) if hasattr(first, "__dataclass_fields__") else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if hasattr(row, "__dataclass_fields__") else row)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">',
        '<feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.14"/>',
        "</filter>",
        "</defs>",
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#111827;font-size:13px}",
        ".title{font-size:22px;font-weight:800}",
        ".sub{fill:#64748b;font-size:12px}",
        ".axis{stroke:#cbd5e1;stroke-width:1}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".label{font-size:12px}",
        ".value{font-size:12px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" rx="18" fill="#ffffff"/>',
    ]


def write_decision_stack_svg(path: Path, metrics: list[dict[str, Any]], title: str) -> None:
    width, height = 980, 470
    left, right, top, bottom = 80, 40, 78, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_total = max(sum(int(row[d]) for d in DECISIONS) for row in metrics) or 1
    bar_w = 96
    gap = (plot_w - bar_w * len(metrics)) / max(1, len(metrics) - 1)
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append('<text class="sub" x="30" y="60">Каждая колонка — слой из того же scan_report.json; цвет — решение слоя.</text>')
    for i in range(6):
        value = round(max_total * i / 5)
        y = top + plot_h - (plot_h * value / max_total)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="sub" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    for idx, row in enumerate(metrics):
        x = left + idx * (bar_w + gap)
        y_cursor = top + plot_h
        for decision in ["ALLOW", "REVIEW", "BLOCK", "MISSING"]:
            value = int(row[decision])
            if value <= 0:
                continue
            h = plot_h * value / max_total
            y_cursor -= h
            lines.append(
                f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" rx="7" fill="{DECISION_COLORS[decision]}" filter="url(#shadow)"/>'
            )
            if h > 20:
                lines.append(f'<text class="value" x="{x+bar_w/2:.1f}" y="{y_cursor+h/2+4:.1f}" text-anchor="middle" fill="#fff">{value}</text>')
        lines.append(f'<text class="label" x="{x+bar_w/2:.1f}" y="{top+plot_h+28}" text-anchor="middle">{html.escape(str(row["subject"]))}</text>')
    legend_x = left
    legend_y = height - 28
    for decision in ["ALLOW", "REVIEW", "BLOCK", "MISSING"]:
        lines.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{DECISION_COLORS[decision]}"/>')
        lines.append(f'<text class="sub" x="{legend_x+20}" y="{legend_y}">{decision}</text>')
        legend_x += 120
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_svg(path: Path, metrics: list[dict[str, Any]], title: str) -> None:
    width, height = 980, 500
    left, right, top, bottom = 74, 38, 80, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    series = [
        ("alert_recall", "Alert recall", "#2563eb"),
        ("block_recall", "BLOCK recall", "#dc2626"),
        ("allow_clean_recall", "Clean ALLOW", "#16a34a"),
    ]
    group_w = plot_w / len(metrics)
    bar_w = 22
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append('<text class="sub" x="30" y="60">Высота — процент. Так видно, кто ловит poisoned, а кто пропускает clean.</text>')
    for i in range(6):
        value = i * 20
        y = top + plot_h - plot_h * value / 100
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="sub" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value}%</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    for idx, row in enumerate(metrics):
        cx = left + group_w * idx + group_w / 2
        for sidx, (key, label, color) in enumerate(series):
            value = float(row[key]) * 100
            h = plot_h * value / 100
            x = cx - (len(series) * bar_w + (len(series) - 1) * 8) / 2 + sidx * (bar_w + 8)
            y = top + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="6" fill="{color}" filter="url(#shadow)"/>')
            lines.append(f'<text class="value" x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{value:.0f}</text>')
        lines.append(f'<text class="label" x="{cx:.1f}" y="{top+plot_h+30}" text-anchor="middle">{html.escape(str(row["subject"]))}</text>')
    legend_x = left
    legend_y = height - 30
    for _, label, color in series:
        lines.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{color}"/>')
        lines.append(f'<text class="sub" x="{legend_x+20}" y="{legend_y}">{html.escape(label)}</text>')
        legend_x += 170
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_risk_svg(path: Path, metrics: list[dict[str, Any]], title: str) -> None:
    width, height = 980, 420
    left, right, top, bottom = 74, 38, 78, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = 80
    gap = (plot_w - bar_w * len(metrics)) / max(1, len(metrics) - 1)
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append('<text class="sub" x="30" y="60">Средний risk score по каждому слою и финальному агрегатору.</text>')
    for i in range(6):
        value = i / 5
        y = top + plot_h - plot_h * value
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="sub" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>')
    for idx, row in enumerate(metrics):
        value = row.get("avg_risk")
        if value is None:
            value = 0.0
            color = "#cbd5e1"
        else:
            color = "#7c3aed" if row["subject"] == "final" else "#0ea5e9"
        h = plot_h * float(value)
        x = left + idx * (bar_w + gap)
        y = top + plot_h - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="8" fill="{color}" filter="url(#shadow)"/>')
        lines.append(f'<text class="value" x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle">{float(value):.2f}</text>')
        lines.append(f'<text class="label" x="{x+bar_w/2:.1f}" y="{top+plot_h+28}" text-anchor="middle">{html.escape(str(row["subject"]))}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_svg(path: Path, rows: list[ReportView], title: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (row.dataset_name, row.dataset_number or 0, row.model_name))
    row_h = 24
    left, top = 260, 92
    cell_w = 112
    label_w = left - 28
    width = left + cell_w * len(SUBJECTS) + 44
    height = top + row_h * len(sorted_rows) + 78
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append('<text class="sub" x="30" y="60">Одна строка — один датасет; четыре клетки — решения sanity/stats/model/final из одного JSON.</text>')
    for idx, subject in enumerate(SUBJECTS):
        x = left + idx * cell_w + cell_w / 2
        lines.append(f'<text class="label" x="{x:.1f}" y="{top-18}" text-anchor="middle">{subject}</text>')
    for ridx, row in enumerate(sorted_rows):
        y = top + ridx * row_h
        expected_color = EXPECTED_COLORS.get(row.expected, EXPECTED_COLORS["unknown"])
        label = row.dataset_id
        if row.model_name:
            label += f" / {row.model_name}"
        lines.append(f'<rect x="28" y="{y+3}" width="10" height="{row_h-6}" rx="2" fill="{expected_color}"/>')
        lines.append(f'<text class="label" x="{label_w}" y="{y+17}" text-anchor="end">{html.escape(label[:36])}</text>')
        for cidx, subject in enumerate(SUBJECTS):
            decision = getattr(row, f"{subject}_decision") if subject != "final" else row.final_decision
            color = DECISION_COLORS.get(decision, DECISION_COLORS["MISSING"])
            x = left + cidx * cell_w + 8
            lines.append(f'<rect x="{x}" y="{y+3}" width="{cell_w-16}" height="{row_h-6}" rx="5" fill="{color}"/>')
            lines.append(f'<text class="value" x="{x+(cell_w-16)/2:.1f}" y="{y+17}" text-anchor="middle" fill="#fff">{decision[0] if decision != "MISSING" else "-"}</text>')
    legend_y = height - 30
    x = 30
    for decision in DECISIONS:
        lines.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{DECISION_COLORS[decision]}"/>')
        lines.append(f'<text class="sub" x="{x+20}" y="{legend_y}">{decision}</text>')
        x += 120
    lines.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{EXPECTED_COLORS["poisoned"]}"/>')
    lines.append(f'<text class="sub" x="{x+20}" y="{legend_y}">poisoned строка</text>')
    x += 150
    lines.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" rx="3" fill="{EXPECTED_COLORS["clean"]}"/>')
    lines.append(f'<text class="sub" x="{x+20}" y="{legend_y}">clean строка</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_top_scanners_svg(path: Path, problems: list[ProblemScanner], title: str, limit: int = 18) -> None:
    counts = Counter(f"{item.layer}: {item.scanner}" for item in problems)
    items = counts.most_common(limit)
    width, height = 1100, 520
    left, right, top, bottom = 72, 40, 76, 150
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max((value for _, value in items), default=1)
    bar_w = min(42, plot_w / max(1, len(items)) * 0.56)
    gap = (plot_w - bar_w * len(items)) / max(1, len(items) - 1)
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append('<text class="sub" x="30" y="60">Какие конкретные проверки чаще всего мешали пройти отчету.</text>')
    for i in range(5):
        value = round(max_value * i / 4)
        y = top + plot_h - plot_h * value / max_value
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="sub" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value}</text>')
    for idx, (name, value) in enumerate(items):
        layer = name.split(":", 1)[0]
        color = {"sanity": "#2563eb", "stats": "#f59e0b", "model": "#7c3aed"}.get(layer, "#64748b")
        h = plot_h * value / max_value
        x = left + idx * (bar_w + gap)
        y = top + plot_h - h
        short = name if len(name) <= 34 else name[:31] + "..."
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="7" fill="{color}" filter="url(#shadow)"/>')
        lines.append(f'<text class="value" x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle">{value}</text>')
        lx = x + bar_w / 2
        ly = top + plot_h + 18
        lines.append(f'<text class="label" x="{lx:.1f}" y="{ly:.1f}" text-anchor="end" transform="rotate(-42 {lx:.1f} {ly:.1f})">{html.escape(short)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_single_flow_svg(path: Path, row: ReportView, title: str) -> None:
    width, height = 1120, 360
    card_w, card_h = 220, 160
    gap = 52
    top = 112
    left = 54
    subjects = [
        ("sanity", "SANITY", row.sanity_decision, row.sanity_risk, row.sanity_bad, row.sanity_scanners, row.sanity_top),
        ("stats", "STATS", row.stats_decision, row.stats_risk, row.stats_bad, row.stats_scanners, row.stats_top),
        ("model", "MODEL", row.model_decision, row.model_risk, row.model_bad, row.model_scanners, row.model_top),
        ("final", "ИТОГ", row.final_decision, row.final_risk, None, None, "aggregator"),
    ]
    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="30" y="38">{html.escape(title)}</text>')
    lines.append(
        f'<text class="sub" x="30" y="62">{html.escape(row.dataset_id)} · expected: {html.escape(row.expected)} · model: {html.escape(row.model_name or "нет")}</text>'
    )
    for idx, (subject, label, decision, risk, bad, total, top_text) in enumerate(subjects):
        x = left + idx * (card_w + gap)
        color = DECISION_COLORS.get(decision, DECISION_COLORS["MISSING"])
        lines.append(f'<rect x="{x}" y="{top}" width="{card_w}" height="{card_h}" rx="18" fill="{color}" filter="url(#shadow)"/>')
        lines.append(f'<text x="{x+24}" y="{top+38}" fill="#fff" font-size="26" font-weight="800">{label}</text>')
        lines.append(f'<text x="{x+24}" y="{top+74}" fill="#fff" font-size="30" font-weight="900">{decision}</text>')
        risk_text = "risk: n/a" if risk is None else f"risk: {risk:.2f}"
        lines.append(f'<text x="{x+24}" y="{top+102}" fill="#fff" font-size="15" font-weight="700">{risk_text}</text>')
        if bad is not None and total is not None:
            lines.append(f'<text x="{x+24}" y="{top+126}" fill="#fff" font-size="14">problem checks: {bad}/{total}</text>')
        if top_text:
            short = top_text if len(top_text) <= 42 else top_text[:39] + "..."
            lines.append(f'<text x="{x+24}" y="{top+148}" fill="#fff" font-size="12">{html.escape(short)}</text>')
        if idx < len(subjects) - 1:
            ax1 = x + card_w + 10
            ax2 = x + card_w + gap - 10
            ay = top + card_h / 2
            lines.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="#64748b" stroke-width="3"/>')
            lines.append(f'<path d="M {ax2} {ay} l -9 -7 v 14 z" fill="#64748b"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def html_table(headers: list[str], rows: list[list[Any]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit is not None else rows
    out = ["<table>", "<thead><tr>"]
    for header in headers:
        out.append(f"<th>{html.escape(str(header))}</th>")
    out.append("</tr></thead><tbody>")
    for row in selected:
        out.append("<tr>")
        for value in row:
            out.append(f"<td>{html.escape(str(value))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    if limit is not None and len(rows) > limit:
        out.append(f"<p class=\"muted\">Показаны первые {limit} строк из {len(rows)}. Полные данные лежат в CSV.</p>")
    return "\n".join(out)


def build_single_dashboard(
    input_path: Path,
    rows: list[ReportView],
    triggers: list[TriggerRow],
    layer_presence: Counter[str],
) -> str:
    row = rows[0]
    trigger_rows = [
        [
            item.layer,
            item.scanner,
            item.status,
            item.category,
            item.subtype,
            item.row_index,
            item.column,
            item.signal,
            item.value,
            item.row_data,
        ]
        for item in triggers
    ]
    status_row = [
        ["Sanity", row.sanity_decision, "" if row.sanity_risk is None else f"{row.sanity_risk:.2f}", f"{row.sanity_bad}/{row.sanity_scanners}"],
        ["Stats", row.stats_decision, "" if row.stats_risk is None else f"{row.stats_risk:.2f}", f"{row.stats_bad}/{row.stats_scanners}"],
        ["Model", row.model_decision, "" if row.model_risk is None else f"{row.model_risk:.2f}", f"{row.model_bad}/{row.model_scanners}"],
        ["Итог", row.final_decision, "" if row.final_risk is None else f"{row.final_risk:.2f}", ""],
    ]
    css = """
    body{margin:0;background:#f6f7fb;color:#111827;font-family:Inter,Segoe UI,Arial,sans-serif}
    .wrap{max-width:1320px;margin:0 auto;padding:28px}
    h1{font-size:30px;margin:0 0 8px}
    h2{font-size:20px;margin:28px 0 12px}
    p{line-height:1.5}
    .muted{color:#64748b}
    .hero{background:white;border:1px solid #e5e7eb;border-radius:18px;padding:18px;box-shadow:0 8px 26px rgba(15,23,42,.06);margin:18px 0}
    .chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
    .chip{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:8px 12px;background:#fff;border:1px solid #e5e7eb;font-weight:700}
    .dot{width:10px;height:10px;border-radius:99px;display:inline-block}
    .chart{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:12px;box-shadow:0 8px 26px rgba(15,23,42,.06)}
    .chart img{display:block;width:100%;height:auto}
    table{border-collapse:collapse;width:100%;background:white;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden}
    th,td{border-bottom:1px solid #e5e7eb;padding:9px 10px;text-align:left;font-size:13px;vertical-align:top}
    th{background:#f8fafc;font-weight:800}
    td:nth-child(8),td:nth-child(9),td:nth-child(10){font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.35}
    code{background:#eef2ff;padding:2px 5px;border-radius:5px}
    .note{background:#fff7ed;border:1px solid #fed7aa;border-radius:14px;padding:12px 14px;margin:16px 0;color:#7c2d12}
    """
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single scan report</title>
<style>{css}</style>
</head>
<body>
<main class="wrap">
<h1>Разбор одного scan_report.json</h1>
<p class="muted">Сформировано: {datetime.now().isoformat(timespec="seconds")} · источник: <code>{html.escape(str(input_path))}</code></p>

<section class="hero">
  <strong>{html.escape(row.dataset_id)}</strong>
  <p class="muted">expected: <code>{html.escape(row.expected)}</code> · model: <code>{html.escape(row.model_name or "нет")}</code> · слои в JSON: <code>{html.escape(str(dict(layer_presence)))}</code></p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{DECISION_COLORS.get(row.sanity_decision, DECISION_COLORS["MISSING"])}"></span>Sanity {row.sanity_decision}</span>
    <span class="chip"><span class="dot" style="background:{DECISION_COLORS.get(row.stats_decision, DECISION_COLORS["MISSING"])}"></span>Stats {row.stats_decision}</span>
    <span class="chip"><span class="dot" style="background:{DECISION_COLORS.get(row.model_decision, DECISION_COLORS["MISSING"])}"></span>Model {row.model_decision}</span>
    <span class="chip"><span class="dot" style="background:{DECISION_COLORS.get(row.final_decision, DECISION_COLORS["MISSING"])}"></span>Итог {row.final_decision}</span>
  </div>
</section>

<h2>Цепочка</h2>
<div class="chart"><img src="charts/single_decision_flow.svg" alt="Цепочка решения одного отчета"></div>

<h2>Статусы слоев</h2>
{html_table(["Слой", "Статус", "Risk", "Проблемных проверок"], status_row)}

<div class="note">Model-слой в этом отчете сказал <strong>{row.model_decision}</strong>. Ниже все равно показаны его suspicious rows: это реальные строки, которые модельные детекторы поднимали как кандидатов, но калибровка оставила слой в ALLOW.</div>

<h2>Реальные данные, на которых сработали проверки</h2>
{html_table(["Слой", "Проверка", "Статус", "Категория", "Подтип", "Строка", "Колонка", "Сигнал", "Значение", "Данные строки"], trigger_rows, limit=220)}

<p class="muted">CSV рядом: <code>trigger_evidence.csv</code>, <code>all_mode_layer_summary.csv</code>.</p>
</main>
</body>
</html>
"""


def build_dashboard(
    input_path: Path,
    output_root: Path,
    rows: list[ReportView],
    problems: list[ProblemScanner],
    metrics: list[dict[str, Any]],
    layer_presence: Counter[str],
    triggers: list[TriggerRow],
) -> str:
    if len(rows) == 1:
        return build_single_dashboard(input_path, rows, triggers, layer_presence)

    expected_counts = Counter(row.expected for row in rows)
    final_counts = decision_counts(rows, "final")
    missing_model = sum(1 for row in rows if row.model_decision == "MISSING")
    durations = [row.duration_sec for row in rows if row.duration_sec is not None]
    avg_duration = mean(durations)

    metric_rows = []
    for item in metrics:
        metric_rows.append(
            [
                item["subject"],
                f"A={item['ALLOW']} / R={item['REVIEW']} / B={item['BLOCK']} / M={item['MISSING']}",
                pct(float(item["alert_precision"])),
                pct(float(item["alert_recall"])),
                pct(float(item["alert_accuracy"])),
                pct(float(item["block_recall"])),
                pct(float(item["allow_clean_recall"])),
                "" if item["avg_risk"] is None else f"{float(item['avg_risk']):.3f}",
            ]
        )

    dataset_rows = []
    for row in sorted(rows, key=lambda value: (value.dataset_name, value.dataset_number or 0, value.model_name)):
        dataset_rows.append(
            [
                row.dataset_id,
                row.expected,
                row.sanity_decision,
                row.stats_decision,
                row.model_decision,
                row.final_decision,
                "" if row.final_risk is None else f"{row.final_risk:.3f}",
                row.sanity_top or row.stats_top or row.model_top,
            ]
        )

    problem_rows = [
        [item.dataset_id, item.expected, item.layer, item.status, item.scanner, item.reason]
        for item in problems
    ]
    single_chart = ""
    if len(rows) == 1:
        row = rows[0]
        single_chart = f"""
<h2>Цепочка решения</h2>
<div class="chart"><img src="charts/single_decision_flow.svg" alt="Цепочка решения одного отчета"></div>
<p class="muted">Для одного JSON важнее смотреть не precision/recall, а вклад слоев: какие слои подняли тревогу, какие промолчали, и как агрегатор сделал final.</p>
<h2>Коротко по этому JSON</h2>
{html_table(["Датасет", "Expected", "Sanity", "Stats", "Model", "Final", "Risk", "Главная причина"], [[row.dataset_id, row.expected, row.sanity_decision, row.stats_decision, row.model_decision, row.final_decision, "" if row.final_risk is None else f"{row.final_risk:.3f}", row.sanity_top or row.stats_top or row.model_top]])}
"""

    css = """
    body{margin:0;background:#f6f7fb;color:#111827;font-family:Inter,Segoe UI,Arial,sans-serif}
    .wrap{max-width:1220px;margin:0 auto;padding:28px}
    h1{font-size:30px;margin:0 0 8px}
    h2{font-size:20px;margin:28px 0 12px}
    p{line-height:1.5}
    .muted{color:#64748b}
    .cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:20px 0}
    .card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:16px;box-shadow:0 8px 26px rgba(15,23,42,.06)}
    .k{font-size:28px;font-weight:800;margin-top:6px}
    .charts{display:grid;grid-template-columns:1fr;gap:18px}
    .chart{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:12px;box-shadow:0 8px 26px rgba(15,23,42,.06)}
    .chart img{display:block;width:100%;height:auto}
    table{border-collapse:collapse;width:100%;background:white;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden}
    th,td{border-bottom:1px solid #e5e7eb;padding:9px 10px;text-align:left;font-size:13px;vertical-align:top}
    th{background:#f8fafc;font-weight:700}
    code{background:#eef2ff;padding:2px 5px;border-radius:5px}
    .pill{display:inline-flex;gap:7px;align-items:center;border:1px solid #e5e7eb;background:white;border-radius:999px;padding:6px 10px;margin:4px 6px 4px 0}
    .dot{width:10px;height:10px;border-radius:99px;display:inline-block}
    """
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>All-mode layer dashboard</title>
<style>{css}</style>
</head>
<body>
<main class="wrap">
<h1>Сравнение слоев внутри одного --mode all</h1>
<p class="muted">Сформировано: {datetime.now().isoformat(timespec="seconds")} · источник: <code>{html.escape(str(input_path))}</code></p>
<div class="cards">
  <div class="card"><div class="muted">Отчетов</div><div class="k">{len(rows)}</div></div>
  <div class="card"><div class="muted">Poisoned / clean</div><div class="k">{expected_counts.get("poisoned",0)} / {expected_counts.get("clean",0)}</div></div>
  <div class="card"><div class="muted">Финальные решения</div><div class="k">A{final_counts.get("ALLOW",0)} R{final_counts.get("REVIEW",0)} B{final_counts.get("BLOCK",0)}</div></div>
  <div class="card"><div class="muted">Среднее время</div><div class="k">{"" if avg_duration is None else f"{avg_duration:.2f}s"}</div></div>
</div>

<p>
  <span class="pill"><span class="dot" style="background:{DECISION_COLORS["ALLOW"]}"></span>ALLOW</span>
  <span class="pill"><span class="dot" style="background:{DECISION_COLORS["REVIEW"]}"></span>REVIEW</span>
  <span class="pill"><span class="dot" style="background:{DECISION_COLORS["BLOCK"]}"></span>BLOCK</span>
  <span class="pill"><span class="dot" style="background:{DECISION_COLORS["MISSING"]}"></span>MISSING</span>
</p>

<p class="muted">Слои внутри JSON: {html.escape(str(dict(layer_presence)))}. Если у слоя стоит MISSING, значит такого detector category нет в этом scan_report.json. В этом наборе model отсутствует в {missing_model} отчетах.</p>

{single_chart}

<section class="charts">
  <div class="chart"><img src="charts/decision_stack.svg" alt="Решения по слоям"></div>
  <div class="chart"><img src="charts/quality_metrics.svg" alt="Метрики качества"></div>
  <div class="chart"><img src="charts/layer_matrix.svg" alt="Матрица решений по датасетам"></div>
  <div class="chart"><img src="charts/risk_by_layer.svg" alt="Risk score по слоям"></div>
  <div class="chart"><img src="charts/top_problem_scanners.svg" alt="Частые проблемные сканеры"></div>
</section>

<h2>Метрики по слоям</h2>
{html_table(["Слой", "Решения", "Alert precision", "Alert recall", "Alert accuracy", "BLOCK recall", "ALLOW clean recall", "Avg risk"], metric_rows)}

<h2>Датасеты: как каждый слой сложился в финал</h2>
{html_table(["Датасет", "Ожидалось", "Sanity", "Stats", "Model", "Final", "Risk", "Главная причина"], dataset_rows, limit=120)}

<h2>Проблемные проверки</h2>
{html_table(["Датасет", "Ожидалось", "Слой", "Статус", "Сканер", "Причина"], problem_rows, limit=160)}

<p class="muted">CSV рядом: <code>all_mode_layer_summary.csv</code>, <code>all_mode_layer_metrics.csv</code>, <code>problem_scanners.csv</code>.</p>
</main>
</body>
</html>
"""


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    index = load_index(Path(args.index).resolve() if args.index else None, args.mode)
    required_layers = set(args.require_layers or [])

    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = output_root / "charts"
    charts_dir.mkdir(exist_ok=True)

    rows: list[ReportView] = []
    problems: list[ProblemScanner] = []
    triggers: list[TriggerRow] = []
    layer_presence: Counter[str] = Counter()
    skipped_by_layer_filter = 0

    for report_path in discover_reports(input_path):
        view, report_problems, present_layers = parse_report(report_path, index)
        if required_layers and not required_layers.issubset(present_layers):
            skipped_by_layer_filter += 1
            continue
        for layer in present_layers:
            layer_presence[layer] += 1
        rows.append(view)
        problems.extend(report_problems)
        triggers.extend(extract_trigger_rows(report_path))

    metrics = build_metrics(rows)
    write_csv(output_root / "all_mode_layer_summary.csv", rows)
    write_csv(output_root / "problem_scanners.csv", problems)
    write_csv(output_root / "all_mode_layer_metrics.csv", metrics)
    write_csv(output_root / "trigger_evidence.csv", triggers)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "input": str(input_path),
                "reports": len(rows),
                "skipped_by_layer_filter": skipped_by_layer_filter,
                "layer_presence": dict(layer_presence),
                "trigger_rows": len(triggers),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_decision_stack_svg(charts_dir / "decision_stack.svg", metrics, "Решения по слоям из одного all-mode прохода")
    write_metrics_svg(charts_dir / "quality_metrics.svg", metrics, "Качество слоев против expected")
    write_matrix_svg(charts_dir / "layer_matrix.svg", rows, "Матрица: слои → финал по каждому датасету")
    write_risk_svg(charts_dir / "risk_by_layer.svg", metrics, "Средний риск по слоям")
    write_top_scanners_svg(charts_dir / "top_problem_scanners.svg", problems, "Частые проблемные проверки")
    if len(rows) == 1:
        write_single_flow_svg(charts_dir / "single_decision_flow.svg", rows[0], "Цепочка решения одного scan_report.json")

    dashboard = build_dashboard(input_path, output_root, rows, problems, metrics, layer_presence, triggers)
    (output_root / "dashboard.html").write_text(dashboard, encoding="utf-8")

    readme = [
        "# All-mode layer analysis",
        "",
        f"Источник: `{input_path}`",
        f"Отчетов: `{len(rows)}`",
        f"Пропущено фильтром слоев: `{skipped_by_layer_filter}`",
        "",
        "Открыть главный отчет: `dashboard.html`.",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Dashboard: {output_root / 'dashboard.html'}")
    print(f"Metrics CSV: {output_root / 'all_mode_layer_metrics.csv'}")
    print(f"Dataset matrix CSV: {output_root / 'all_mode_layer_summary.csv'}")
    if skipped_by_layer_filter:
        print(f"Skipped by layer filter: {skipped_by_layer_filter}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сравнить sanity/stats/model/final внутри одного --mode all scan_report.json.")
    parser.add_argument("--input", required=True, help="Папка с scan_report.json или один JSON.")
    parser.add_argument("--output", required=True, help="Папка вывода dashboard/CSV/SVG.")
    parser.add_argument("--index", default=None, help="Опциональный index.json раннера для expected/duration/model metadata.")
    parser.add_argument("--mode", default=None, help="Если указан index.json, брать metadata только для этого mode.")
    parser.add_argument("--require-layers", nargs="*", choices=LAYERS, default=None, help="Оставить только отчеты, где есть все перечисленные слои.")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

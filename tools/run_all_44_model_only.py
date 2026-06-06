from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DATASET_RE = re.compile(r"dataset_(\d+)\.(csv|jsonl|parquet)$", re.IGNORECASE)

TABULAR_CONFIGS = {
    "bank_churn": {"target": "Exited", "drop": ["Complain"]},
    "creditcard_fraud_ulb": {"target": "Class", "drop": ["Time"]},
    "german_credit": {"target": "Risk", "drop": []},
    "give_me_some_credit": {"target": "SeriousDlqin2yrs", "drop": []},
    "taiwan_credit_default": {"target": "default.payment.next.month", "drop": []},
    "uci_banknote_auth": {"target": "label", "drop": []},
}

NLP_DATASETS = {
    "banking77",
    "bitext_retail_banking",
    "fingpt_sentiment",
    "fin_phrasebank",
    "twitter_fin_sentiment",
}

MODEL_PREFERENCES = ("distilbert", "finbert")


@dataclass
class Job:
    dataset_name: str
    dataset_number: int
    dataset_path: Path
    expected: str
    command: list[str]
    case_name: str
    model_name: str | None = None


@dataclass
class Result:
    dataset_name: str
    dataset_number: int
    dataset_path: str
    expected: str
    case_name: str
    model_name: str | None
    command: str
    return_code: int | None
    duration_sec: float
    report_path: str | None
    copied_report_path: str | None
    stdout_path: str
    stderr_path: str
    decision: str | None
    risk_score: float | None
    overall_status: str | None
    error: str | None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def dataset_number(path: Path) -> int:
    match = DATASET_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Bad dataset filename: {path}")
    return int(match.group(1))


def expected_by_number(number: int) -> str:
    return "clean" if number == 4 else "poisoned"


def reports_snapshot(root: Path) -> set[Path]:
    reports_dir = root / "reports"
    if not reports_dir.exists():
        return set()
    return {path.resolve() for path in reports_dir.glob("scan_report_*.json")}


def find_new_report(root: Path, before: set[Path]) -> Path | None:
    reports_dir = root / "reports"
    if not reports_dir.exists():
        return None

    after = {path.resolve() for path in reports_dir.glob("scan_report_*.json")}
    new_reports = sorted(after - before, key=lambda path: path.stat().st_mtime)
    return new_reports[-1] if new_reports else None


def extract_summary(report_path: Path | None) -> tuple[str | None, float | None, str | None]:
    if report_path is None or not report_path.exists():
        return None, None, None

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None

    overall_status = report.get("overall_status")
    final = (report.get("metadata") or {}).get("final_decision") or {}
    if isinstance(final, dict) and final:
        decision = final.get("decision")
        risk = final.get("risk_score")
        try:
            risk = float(risk) if risk is not None else None
        except Exception:
            risk = None
        return str(decision).upper() if decision else None, risk, overall_status

    if overall_status == "passed":
        return "ALLOW", None, overall_status
    if overall_status == "hand_check":
        return "REVIEW", None, overall_status
    if overall_status == "failed":
        return "BLOCK", None, overall_status

    return None, None, overall_status


def choose_model(root: Path, dataset_name: str, number: int, family: str) -> tuple[Path | None, Path | None, str | None]:
    model_dir = root / "models" / dataset_name
    if not model_dir.exists():
        return None, None, None

    families = MODEL_PREFERENCES if family == "auto" else (family,)
    for prefix in families:
        candidate_name = f"{prefix}_clean" if number == 4 else f"{prefix}_dataset_{number}"
        reference_name = f"{prefix}_clean"
        candidate = model_dir / candidate_name / "model.safetensors"
        reference = model_dir / reference_name / "model.safetensors"
        if candidate.exists() and reference.exists():
            return candidate, reference, candidate_name

    return None, None, None


def add_scan_mode(command: list[str], scan_mode: str) -> None:
    if scan_mode == "model":
        command.extend(["--mode", "only", "--only", "model"])
    else:
        command.extend(["--mode", "all"])


def build_tabular_job(
    root: Path,
    python_exe: str,
    dataset_path: Path,
    use_reference: bool,
    scan_mode: str,
) -> Job:
    name = dataset_path.parent.name
    number = dataset_number(dataset_path)
    cfg = TABULAR_CONFIGS[name]

    command = [
        python_exe,
        "-m",
        "src.cli",
        "scan",
        "--dataset",
        rel(dataset_path, root),
        "--target",
        cfg["target"],
    ]

    if cfg["drop"]:
        command.extend(["--drop", ",".join(cfg["drop"])])

    if use_reference:
        command.extend(["--reference", f"data/{name}/dataset_4.csv"])

    add_scan_mode(command, scan_mode)

    return Job(
        dataset_name=name,
        dataset_number=number,
        dataset_path=dataset_path,
        expected=expected_by_number(number),
        command=command,
        case_name=f"{name}_dataset_{number}",
        model_name="auto_mlp",
    )


def build_nlp_job(
    root: Path,
    python_exe: str,
    dataset_path: Path,
    family: str,
    use_reference: bool,
    scan_mode: str,
) -> Job:
    name = dataset_path.parent.name
    number = dataset_number(dataset_path)
    model, reference, model_name = choose_model(root, name, number, family)

    command = [
        python_exe,
        "-m",
        "src.cli",
        "scan",
        "--dataset",
        rel(dataset_path, root),
    ]

    if model:
        command.extend(["--model", rel(model, root)])

    if use_reference and reference:
        command.extend(["--reference", rel(reference, root)])

    add_scan_mode(command, scan_mode)

    return Job(
        dataset_name=name,
        dataset_number=number,
        dataset_path=dataset_path,
        expected=expected_by_number(number),
        command=command,
        case_name=f"{name}_dataset_{number}",
        model_name=model_name,
    )


def build_jobs(root: Path, python_exe: str, family: str, use_reference: bool, scan_mode: str) -> list[Job]:
    data_dir = root / "data"
    jobs: list[Job] = []

    dataset_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and DATASET_RE.fullmatch(path.name)
    )

    for dataset_path in dataset_paths:
        name = dataset_path.parent.name
        if name in TABULAR_CONFIGS:
            jobs.append(build_tabular_job(root, python_exe, dataset_path, use_reference, scan_mode))
        elif name in NLP_DATASETS:
            jobs.append(build_nlp_job(root, python_exe, dataset_path, family, use_reference, scan_mode))
        else:
            raise ValueError(f"No runner config for dataset folder: {name}")

    return jobs


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_one(root: Path, output_root: Path, job: Job, timeout: int, dry_run: bool) -> Result:
    target_dir = output_root / safe_name(job.dataset_name) / f"dataset_{job.dataset_number}"
    target_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = target_dir / "stdout.txt"
    stderr_path = target_dir / "stderr.txt"
    copied_report_path = target_dir / "scan_report.json"
    command = command_text(job.command)

    if dry_run:
        stdout_path.write_text(command + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return Result(
            dataset_name=job.dataset_name,
            dataset_number=job.dataset_number,
            dataset_path=rel(job.dataset_path, root),
            expected=job.expected,
            case_name=job.case_name,
            model_name=job.model_name,
            command=command,
            return_code=None,
            duration_sec=0.0,
            report_path=None,
            copied_report_path=None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            decision=None,
            risk_score=None,
            overall_status=None,
            error="dry_run",
        )

    if job.dataset_name in NLP_DATASETS and not job.model_name:
        message = f"No matching NLP model found for {job.dataset_name} dataset_{job.dataset_number}"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(message + "\n", encoding="utf-8")
        return Result(
            dataset_name=job.dataset_name,
            dataset_number=job.dataset_number,
            dataset_path=rel(job.dataset_path, root),
            expected=job.expected,
            case_name=job.case_name,
            model_name=None,
            command=command,
            return_code=None,
            duration_sec=0.0,
            report_path=None,
            copied_report_path=None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            decision=None,
            risk_score=None,
            overall_status=None,
            error="missing_model",
        )

    before = reports_snapshot(root)
    started = time.time()

    try:
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        completed = subprocess.run(
            job.command,
            cwd=root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        duration = round(time.time() - started, 3)

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        report_path = find_new_report(root, before)
        copied_report = None
        if report_path and report_path.exists():
            shutil.copy2(report_path, copied_report_path)
            copied_report = str(copied_report_path)

        decision, risk, overall_status = extract_summary(report_path)
        return Result(
            dataset_name=job.dataset_name,
            dataset_number=job.dataset_number,
            dataset_path=rel(job.dataset_path, root),
            expected=job.expected,
            case_name=job.case_name,
            model_name=job.model_name,
            command=command,
            return_code=completed.returncode,
            duration_sec=duration,
            report_path=str(report_path) if report_path else None,
            copied_report_path=copied_report,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            decision=decision,
            risk_score=risk,
            overall_status=overall_status,
            error=None if report_path else "report_not_found",
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.time() - started, 3)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return Result(
            dataset_name=job.dataset_name,
            dataset_number=job.dataset_number,
            dataset_path=rel(job.dataset_path, root),
            expected=job.expected,
            case_name=job.case_name,
            model_name=job.model_name,
            command=command,
            return_code=-1,
            duration_sec=duration,
            report_path=None,
            copied_report_path=None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            decision=None,
            risk_score=None,
            overall_status=None,
            error=f"timeout after {timeout}s",
        )


def write_index(output_root: Path, results: list[Result]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_root / "index.csv"
    if not results:
        csv_path.write_text("", encoding="utf-8")
        return

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scans for all 44 datasets.")
    parser.add_argument("--python", default="python3", help="Python executable inside the activated venv.")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per dataset, seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N jobs.")
    parser.add_argument("--start-at", type=int, default=1, help="Start from 1-based job index.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands without executing them.")
    parser.add_argument("--no-reference", action="store_true", help="Mode 1 triage: do not pass clean references.")
    parser.add_argument(
        "--scan-mode",
        choices=["all", "model"],
        default="all",
        help="all = run every scanner layer; model = run only the model/post-train layer.",
    )
    parser.add_argument(
        "--nlp-family",
        choices=["auto", "distilbert", "finbert"],
        default="auto",
        help="NLP model family. auto prefers distilbert, then finbert.",
    )
    args = parser.parse_args()

    root = project_root()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "all_44_model_only" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        root=root,
        python_exe=args.python,
        family=args.nlp_family,
        use_reference=not args.no_reference,
        scan_mode=args.scan_mode,
    )
    if args.start_at < 1:
        parser.error("--start-at must be >= 1")

    jobs = jobs[args.start_at - 1 :]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Project root: {root}")
    print(f"Output root: {output_root}")
    print(f"Jobs: {len(jobs)}")
    print(f"Start at: {args.start_at}")
    print(f"Mode: {'mode1/no-reference' if args.no_reference else 'mode2/reference'}")
    print(f"Scan mode: {args.scan_mode}")
    print(f"NLP family: {args.nlp_family}")

    results: list[Result] = []
    try:
        for index, job in enumerate(jobs, start=1):
            print(f"\n[{index}/{len(jobs)}] {job.case_name}")
            print(f"COMMAND: {command_text(job.command)}")
            result = run_one(root, output_root, job, args.timeout, args.dry_run)
            results.append(result)
            write_index(output_root, results)
            print(
                "  "
                f"rc={result.return_code} "
                f"decision={result.decision} "
                f"risk={result.risk_score} "
                f"status={result.overall_status} "
                f"time={result.duration_sec}s"
            )
            if result.error:
                print(f"  error={result.error}")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial index...")

    write_index(output_root, results)
    print("\nDONE")
    print(f"Index CSV: {output_root / 'index.csv'}")
    print(f"Index JSON: {output_root / 'index.json'}")
    print(f"Reports: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

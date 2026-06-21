from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DATASET_RE = re.compile(r"dataset_(\d+)\.(csv|jsonl|parquet)$", re.IGNORECASE)


@dataclass
class Job:
    dataset_name: str
    dataset_number: int
    dataset_path: Path
    model_name: str
    model_path: Path


@dataclass
class Result:
    dataset_name: str
    dataset_number: int
    dataset_path: str
    model_name: str
    model_path: str
    expected: str
    return_code: int
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
    return str(path.resolve().relative_to(root.resolve()))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def get_dataset_number(path: Path) -> int:
    match = DATASET_RE.match(path.name)
    if not match:
        raise ValueError(f"Bad dataset filename: {path.name}")
    return int(match.group(1))


def expected_by_number(number: int) -> str:
    return "clean" if number == 4 else "poisoned"


def list_datasets(root: Path) -> list[Path]:
    data_dir = root / "data"
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and DATASET_RE.match(path.name)
    )


def list_models_for_dataset(root: Path, dataset_name: str) -> list[Path]:
    model_dir = root / "models" / dataset_name
    if not model_dir.exists():
        return []

    return sorted(model_dir.glob("*/model.safetensors"))


def list_matching_models_for_dataset(root: Path, dataset_name: str, number: int) -> list[Path]:
    model_dir = root / "models" / dataset_name
    if not model_dir.exists():
        return []

    if number == 4:
        return sorted(model_dir.glob("*_clean/model.safetensors"))

    return sorted(model_dir.glob(f"*_dataset_{number}/model.safetensors"))


def build_jobs(root: Path, pairing: str) -> list[Job]:
    jobs: list[Job] = []

    for dataset_path in list_datasets(root):
        dataset_name = dataset_path.parent.name
        number = get_dataset_number(dataset_path)

        if pairing == "matching":
            model_paths = list_matching_models_for_dataset(root, dataset_name, number)
        else:
            model_paths = list_models_for_dataset(root, dataset_name)

        for model_path in model_paths:
            jobs.append(
                Job(
                    dataset_name=dataset_name,
                    dataset_number=number,
                    dataset_path=dataset_path,
                    model_name=model_path.parent.name,
                    model_path=model_path,
                )
            )

    return jobs


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

    if new_reports:
        return new_reports[-1]

    all_reports = sorted(reports_dir.glob("scan_report_*.json"), key=lambda path: path.stat().st_mtime)
    return all_reports[-1] if all_reports else None


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

    for item in report.get("results", []):
        details = item.get("details") or {}
        if "decision" in details:
            decision = details.get("decision")
            risk = details.get("risk_score")
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


def run_one(
    root: Path,
    python_exe: str,
    output_root: Path,
    job: Job,
    timeout: int,
) -> Result:
    target_dir = (
        output_root
        / safe_name(job.dataset_name)
        / f"dataset_{job.dataset_number}"
        / safe_name(job.model_name)
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = target_dir / "stdout.txt"
    stderr_path = target_dir / "stderr.txt"
    copied_report_path = target_dir / "scan_report.json"

    command = [
        python_exe,
        "-m",
        "src.cli",
        "scan",
        "--dataset",
        rel(job.dataset_path, root),
        "--model",
        rel(job.model_path, root),
        "--mode",
        "all",
    ]

    print("COMMAND:", " ".join(f'"{x}"' if " " in x else x for x in command))

    before = reports_snapshot(root)
    started = time.time()

    try:
        completed = subprocess.run(
            command,
            cwd=root,
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
            model_name=job.model_name,
            model_path=rel(job.model_path, root),
            expected=expected_by_number(job.dataset_number),
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
            model_name=job.model_name,
            model_path=rel(job.model_path, root),
            expected=expected_by_number(job.dataset_number),
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

    json_path = output_root / "index.json"
    csv_path = output_root / "index.csv"

    json_path.write_text(
        json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not results:
        csv_path.write_text("", encoding="utf-8")
        return

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--pairing",
        choices=["all_models", "matching"],
        default="all_models",
        help="all_models = every dataset with every model in same model folder; matching = dataset_N with *_dataset_N, dataset_4 with *_clean",
    )

    args = parser.parse_args()

    root = project_root()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "real_scan_reports" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(root, args.pairing)

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Project root: {root}")
    print(f"Output root: {output_root}")
    print(f"Jobs: {len(jobs)}")
    print(f"Pairing: {args.pairing}")

    results: list[Result] = []

    try:
        for i, job in enumerate(jobs, start=1):
            print(
                f"\n[{i}/{len(jobs)}] "
                f"dataset={rel(job.dataset_path, root)} "
                f"model={rel(job.model_path, root)}"
            )

            result = run_one(
                root=root,
                python_exe=args.python,
                output_root=output_root,
                job=job,
                timeout=args.timeout,
            )

            results.append(result)
            write_index(output_root, results)

            print(
                f"  rc={result.return_code} "
                f"decision={result.decision} "
                f"risk={result.risk_score} "
                f"status={result.overall_status} "
                f"time={result.duration_sec}s"
            )
            print(f"  report={result.copied_report_path}")

            if result.error:
                print(f"  error={result.error}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial index...")

    write_index(output_root, results)

    print(f"\nDONE")
    print(f"Index CSV: {output_root / 'index.csv'}")
    print(f"Index JSON: {output_root / 'index.json'}")
    print(f"Reports: {output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
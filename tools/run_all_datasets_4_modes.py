from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET_RE = re.compile(r"dataset_(\d+)\.(csv|jsonl|parquet)$", re.IGNORECASE)
MODES = ("sanity", "stats", "model", "all")
MODEL_PREFERENCES = ("distilbert", "finbert")

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

LABEL_EXACT = (
    "label",
    "target",
    "y",
    "class",
    "Class",
    "Risk",
    "Exited",
    "SeriousDlqin2yrs",
    "default.payment.next.month",
)

LABEL_KEYWORDS = (
    "label",
    "target",
    "class",
    "risk",
    "exited",
    "default",
    "fraud",
    "churn",
    "sentiment",
    "intent",
    "category",
)


@dataclass
class DatasetInfo:
    dataset_name: str
    dataset_number: int
    dataset_path: Path
    expected: str
    target: str | None
    drop: list[str]
    is_tabular: bool
    is_nlp: bool


@dataclass
class Job:
    index: int
    mode: str
    info: DatasetInfo
    command: list[str]
    model_name: str | None = None
    error: str | None = None


@dataclass
class Result:
    job_index: int
    mode: str
    dataset_name: str
    dataset_number: int
    dataset_path: str
    expected: str
    target: str | None
    drop: str
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


def path_arg(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def dataset_number(path: Path) -> int:
    match = DATASET_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Bad dataset filename: {path}")
    return int(match.group(1))


def expected_by_number(number: int, clean_number: int) -> str:
    return "clean" if number == clean_number else "poisoned"


def read_sample(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=1000)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def infer_target(df) -> str | None:
    for column in LABEL_EXACT:
        if column in df.columns:
            return column

    for column in df.columns:
        norm = re.sub(r"[^a-z]", "", str(column).lower())
        for keyword in LABEL_KEYWORDS:
            if norm == keyword or norm.startswith(keyword):
                return str(column)
    return None


def has_long_text(df) -> bool:
    for column in df.columns:
        series = df[column].dropna()
        if len(series) == 0:
            continue
        if series.dtype == object:
            try:
                if float(series.astype(str).str.len().mean()) >= 25:
                    return True
            except Exception:
                continue
    return False


def inspect_dataset(path: Path, clean_number: int) -> DatasetInfo:
    dataset_name = path.parent.name
    number = dataset_number(path)
    config = TABULAR_CONFIGS.get(dataset_name, {})
    target = config.get("target")
    drop = list(config.get("drop", []))
    is_known_nlp = dataset_name in NLP_DATASETS

    is_tabular = False
    is_nlp = is_known_nlp

    if target is not None:
        is_tabular = not is_known_nlp
    else:
        try:
            df = read_sample(path)
            target = infer_target(df)
            text_like = has_long_text(df)
            numeric_features = 0
            if target and target in df.columns:
                numeric_features = int(
                    df.drop(columns=[target, *drop], errors="ignore")
                    .select_dtypes(include="number")
                    .shape[1]
                )
            is_nlp = is_known_nlp or (text_like and numeric_features == 0)
            is_tabular = bool(target and numeric_features > 0 and not is_nlp)
        except Exception:
            pass

    return DatasetInfo(
        dataset_name=dataset_name,
        dataset_number=number,
        dataset_path=path,
        expected=expected_by_number(number, clean_number),
        target=target,
        drop=drop,
        is_tabular=is_tabular,
        is_nlp=is_nlp,
    )


def list_datasets(data_root: Path) -> list[Path]:
    return sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and DATASET_RE.fullmatch(path.name)
    )


def find_clean_dataset(info: DatasetInfo, clean_number: int) -> Path | None:
    clean = info.dataset_path.with_name(f"dataset_{clean_number}{info.dataset_path.suffix}")
    return clean if clean.exists() else None


def choose_model(root: Path, dataset_name: str, number: int, family: str, clean_number: int):
    model_dir = root / "models" / dataset_name
    if not model_dir.exists():
        return None, None, None

    families = MODEL_PREFERENCES if family == "auto" else (family,)
    for prefix in families:
        candidate_name = f"{prefix}_clean" if number == clean_number else f"{prefix}_dataset_{number}"
        reference_name = f"{prefix}_clean"
        candidate = model_dir / candidate_name / "model.safetensors"
        reference = model_dir / reference_name / "model.safetensors"
        if candidate.exists():
            return candidate, reference if reference.exists() else None, candidate_name

    if number == clean_number:
        candidates = sorted(model_dir.glob("*_clean/model.safetensors"))
    else:
        candidates = sorted(model_dir.glob(f"*_dataset_{number}/model.safetensors"))
    if not candidates:
        return None, None, None

    candidate = candidates[0]
    prefix = candidate.parent.name.split("_dataset_", 1)[0].removesuffix("_clean")
    reference = model_dir / f"{prefix}_clean" / "model.safetensors"
    return candidate, reference if reference.exists() else None, candidate.parent.name


def build_command(
    root: Path,
    python_exe: str,
    info: DatasetInfo,
    mode: str,
    family: str,
    clean_number: int,
    use_reference: bool,
) -> tuple[list[str], str | None, str | None]:
    command = [
        python_exe,
        "-m",
        "src.cli",
        "scan",
        "--dataset",
        path_arg(info.dataset_path, root),
    ]
    model_name = None

    if mode in {"model", "all"}:
        if info.is_tabular:
            if not info.target:
                return command, None, "missing_target"
            command.extend(["--target", info.target])
            if info.drop:
                command.extend(["--drop", ",".join(info.drop)])
            clean_dataset = find_clean_dataset(info, clean_number)
            if use_reference and clean_dataset:
                command.extend(["--reference", path_arg(clean_dataset, root)])
            model_name = "auto_mlp"
        else:
            model, reference, model_name = choose_model(
                root, info.dataset_name, info.dataset_number, family, clean_number
            )
            if model is None:
                return command, None, "missing_model"
            command.extend(["--model", path_arg(model, root)])
            if use_reference and reference is not None:
                command.extend(["--reference", path_arg(reference, root)])

    if mode == "all":
        command.extend(["--mode", "all"])
    else:
        command.extend(["--mode", "only", "--only", mode])

    return command, model_name, None


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


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def result_paths(output_root: Path, job: Job) -> tuple[Path, Path, Path, Path]:
    info = job.info
    target_dir = (
        output_root
        / safe_name(job.mode)
        / safe_name(info.dataset_name)
        / f"dataset_{info.dataset_number}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    return (
        target_dir,
        target_dir / "stdout.txt",
        target_dir / "stderr.txt",
        target_dir / "scan_report.json",
    )


def make_result(
    root: Path,
    output_root: Path,
    job: Job,
    return_code: int | None,
    duration: float,
    report_path: Path | None,
    copied_report_path: Path | None,
    stdout_path: Path,
    stderr_path: Path,
    error: str | None,
) -> Result:
    decision, risk, overall_status = extract_summary(report_path)
    info = job.info
    return Result(
        job_index=job.index,
        mode=job.mode,
        dataset_name=info.dataset_name,
        dataset_number=info.dataset_number,
        dataset_path=path_arg(info.dataset_path, root),
        expected=info.expected,
        target=info.target,
        drop=",".join(info.drop),
        model_name=job.model_name,
        command=command_text(job.command),
        return_code=return_code,
        duration_sec=round(duration, 3),
        report_path=str(report_path) if report_path else None,
        copied_report_path=str(copied_report_path) if copied_report_path else None,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        decision=decision,
        risk_score=risk,
        overall_status=overall_status,
        error=error,
    )


def run_one(root: Path, output_root: Path, job: Job, timeout: int, dry_run: bool) -> Result:
    _, stdout_path, stderr_path, copied_report_path = result_paths(output_root, job)

    if dry_run or job.error:
        stdout_path.write_text(command_text(job.command) + "\n", encoding="utf-8")
        stderr_path.write_text((job.error or "dry_run") + "\n", encoding="utf-8")
        return make_result(
            root=root,
            output_root=output_root,
            job=job,
            return_code=None,
            duration=0.0,
            report_path=None,
            copied_report_path=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error=job.error or "dry_run",
        )

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    before = reports_snapshot(root)
    started = time.time()
    try:
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
        duration = time.time() - started
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        report_path = find_new_report(root, before)
        copied = None
        error = None if report_path else "report_not_found"
        if report_path:
            shutil.copy2(report_path, copied_report_path)
            copied = copied_report_path
        return make_result(
            root,
            output_root,
            job,
            completed.returncode,
            duration,
            report_path,
            copied,
            stdout_path,
            stderr_path,
            error,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return make_result(
            root,
            output_root,
            job,
            -1,
            duration,
            None,
            None,
            stdout_path,
            stderr_path,
            f"timeout after {timeout}s",
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


def build_jobs(
    root: Path,
    data_root: Path,
    python_exe: str,
    modes: list[str],
    family: str,
    clean_number: int,
    use_reference: bool,
) -> list[Job]:
    infos = [inspect_dataset(path, clean_number) for path in list_datasets(data_root)]
    jobs: list[Job] = []
    job_index = 1
    for info in infos:
        for mode in modes:
            command, model_name, error = build_command(
                root=root,
                python_exe=python_exe,
                info=info,
                mode=mode,
                family=family,
                clean_number=clean_number,
                use_reference=use_reference,
            )
            jobs.append(Job(job_index, mode, info, command, model_name, error))
            job_index += 1
    return jobs


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip().lower() for item in raw.split(",") if item.strip()]
    bad = [mode for mode in modes if mode not in MODES]
    if bad:
        raise argparse.ArgumentTypeError(f"bad mode(s): {', '.join(bad)}")
    return modes or list(MODES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every discovered dataset in four scanner modes.")
    parser.add_argument("--python", default="python3", help="Python executable to run src.cli.")
    parser.add_argument("--data-root", default="data", help="Dataset root. Can point to a 140-dataset folder.")
    parser.add_argument("--modes", type=parse_modes, default=list(MODES), help="Comma list: sanity,stats,model,all.")
    parser.add_argument("--nlp-family", choices=["auto", "distilbert", "finbert"], default="auto")
    parser.add_argument("--clean-number", type=int, default=4, help="dataset_N number that is clean.")
    parser.add_argument("--no-reference", action="store_true", help="Do not pass clean references.")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per job, seconds.")
    parser.add_argument("--start-at", type=int, default=1, help="Start from 1-based job index.")
    parser.add_argument("--limit", type=int, default=0, help="Run only N jobs after --start-at.")
    parser.add_argument("--dry-run", action="store_true", help="Build commands without running them.")
    args = parser.parse_args()

    root = project_root()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    data_root = data_root.resolve()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "all_datasets_4_modes" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    all_jobs = build_jobs(
        root=root,
        data_root=data_root,
        python_exe=args.python,
        modes=args.modes,
        family=args.nlp_family,
        clean_number=args.clean_number,
        use_reference=not args.no_reference,
    )

    if args.start_at < 1:
        parser.error("--start-at must be >= 1")
    jobs = all_jobs[args.start_at - 1 :]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Project root: {root}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Datasets: {len(list_datasets(data_root))}")
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Selected jobs: {len(jobs)}")
    print(f"Modes: {','.join(args.modes)}")
    print(f"Clean number: {args.clean_number}")
    print(f"References: {'off' if args.no_reference else 'on'}")

    results: list[Result] = []
    try:
        for job in jobs:
            info = job.info
            print(
                f"\n[{job.index}/{len(all_jobs)}] "
                f"{job.mode} {info.dataset_name}/dataset_{info.dataset_number}"
            )
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

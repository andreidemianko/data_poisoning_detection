from __future__ import annotations

import argparse
from pathlib import Path

import run_all_datasets_4_modes as base


PROFILE_DEFS = {
    "sanity": {"cli_mode": "only", "only": "sanity", "reference": False, "needs_model": False},
    "stats": {"cli_mode": "only", "only": "stats", "reference": False, "needs_model": False},
    "model_m1": {"cli_mode": "only", "only": "model", "reference": False, "needs_model": True},
    "model_m2": {"cli_mode": "only", "only": "model", "reference": True, "needs_model": True},
    "default_m1": {"cli_mode": "default", "only": None, "reference": False, "needs_model": True},
    "default_m2": {"cli_mode": "default", "only": None, "reference": True, "needs_model": True},
    "all_m1": {"cli_mode": "all", "only": None, "reference": False, "needs_model": True},
    "all_m2": {"cli_mode": "all", "only": None, "reference": True, "needs_model": True},
}

PRESETS = {
    # Seven-profile set: two data-only layers, model mode1/mode2,
    # default mode1/mode2, and the main production all-mode with reference.
    "seven": ["sanity", "stats", "model_m1", "model_m2", "default_m1", "default_m2", "all_m2"],
    # Full matrix: includes all_m1 too.
    "full": ["sanity", "stats", "model_m1", "model_m2", "default_m1", "default_m2", "all_m1", "all_m2"],
}


def parse_profiles(raw: str) -> list[str]:
    profiles = [item.strip().lower() for item in raw.split(",") if item.strip()]
    bad = [profile for profile in profiles if profile not in PROFILE_DEFS]
    if bad:
        raise argparse.ArgumentTypeError(f"bad profile(s): {', '.join(bad)}")
    return profiles


def build_profile_command(
    root: Path,
    python_exe: str,
    info: base.DatasetInfo,
    profile: str,
    family: str,
    clean_number: int,
) -> tuple[list[str], str | None, str | None]:
    definition = PROFILE_DEFS[profile]
    command = [
        python_exe,
        "-m",
        "src.cli",
        "scan",
        "--dataset",
        base.path_arg(info.dataset_path, root),
    ]
    model_name = None

    if definition["needs_model"]:
        if info.is_tabular:
            if not info.target:
                return command, None, "missing_target"
            command.extend(["--target", info.target])
            if info.drop:
                command.extend(["--drop", ",".join(info.drop)])
            if definition["reference"]:
                clean_dataset = base.find_clean_dataset(info, clean_number)
                if clean_dataset:
                    command.extend(["--reference", base.path_arg(clean_dataset, root)])
            model_name = "auto_mlp"
        else:
            model, reference, model_name = base.choose_model(
                root, info.dataset_name, info.dataset_number, family, clean_number
            )
            if model is None:
                return command, None, "missing_model"
            command.extend(["--model", base.path_arg(model, root)])
            if definition["reference"] and reference is not None:
                command.extend(["--reference", base.path_arg(reference, root)])

    cli_mode = definition["cli_mode"]
    if cli_mode == "only":
        command.extend(["--mode", "only", "--only", str(definition["only"])])
    else:
        command.extend(["--mode", cli_mode])

    return command, model_name, None


def build_jobs(
    root: Path,
    data_root: Path,
    python_exe: str,
    profiles: list[str],
    family: str,
    clean_number: int,
) -> list[base.Job]:
    infos = [base.inspect_dataset(path, clean_number) for path in base.list_datasets(data_root)]
    jobs: list[base.Job] = []
    index = 1
    for info in infos:
        for profile in profiles:
            command, model_name, error = build_profile_command(
                root=root,
                python_exe=python_exe,
                info=info,
                profile=profile,
                family=family,
                clean_number=clean_number,
            )
            jobs.append(base.Job(index, profile, info, command, model_name, error))
            index += 1
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every dataset through named scan profiles.")
    parser.add_argument("--python", default="python3", help="Python executable to run src.cli.")
    parser.add_argument("--data-root", default="data", help="Dataset root. Can point to a 140-dataset folder.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="seven")
    parser.add_argument("--profiles", type=parse_profiles, default=None, help="Comma list overriding --preset.")
    parser.add_argument("--nlp-family", choices=["auto", "distilbert", "finbert"], default="auto")
    parser.add_argument("--clean-number", type=int, default=4, help="dataset_N number that is clean.")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per job, seconds.")
    parser.add_argument("--start-at", type=int, default=1, help="Start from 1-based job index.")
    parser.add_argument("--limit", type=int, default=0, help="Run only N jobs after --start-at.")
    parser.add_argument("--dry-run", action="store_true", help="Build commands without running them.")
    args = parser.parse_args()

    root = base.project_root()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    data_root = data_root.resolve()

    profiles = args.profiles if args.profiles is not None else PRESETS[args.preset]
    run_id = base.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = root / "outputs" / "all_datasets_profiles" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    all_jobs = build_jobs(
        root=root,
        data_root=data_root,
        python_exe=args.python,
        profiles=profiles,
        family=args.nlp_family,
        clean_number=args.clean_number,
    )

    if args.start_at < 1:
        parser.error("--start-at must be >= 1")
    jobs = all_jobs[args.start_at - 1 :]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Project root: {root}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Datasets: {len(base.list_datasets(data_root))}")
    print(f"Profiles: {','.join(profiles)}")
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Selected jobs: {len(jobs)}")
    print(f"Clean number: {args.clean_number}")

    results: list[base.Result] = []
    try:
        for job in jobs:
            info = job.info
            print(
                f"\n[{job.index}/{len(all_jobs)}] "
                f"{job.mode} {info.dataset_name}/dataset_{info.dataset_number}"
            )
            print(f"COMMAND: {base.command_text(job.command)}")
            result = base.run_one(root, output_root, job, args.timeout, args.dry_run)
            results.append(result)
            base.write_index(output_root, results)
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

    base.write_index(output_root, results)
    print("\nDONE")
    print(f"Index CSV: {output_root / 'index.csv'}")
    print(f"Index JSON: {output_root / 'index.json'}")
    print(f"Reports: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

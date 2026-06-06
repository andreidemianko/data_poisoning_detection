import typer

from src.core.factory import discover_scanners, registry
from src.core.pipeline import ExecutionPlan, SecurityPipeline
from src.core.demo import init_demo_assets
from src.scanners.base import ScannerCategory

app = typer.Typer(add_completion=False)


def _build_plan(mode: str, only: str, continue_on_fail: bool) -> ExecutionPlan:
    if mode == "only":
        category = ScannerCategory(only)
        return ExecutionPlan(categories=[category], continue_on_fail=continue_on_fail)

    categories = [ScannerCategory.SANITY, ScannerCategory.STATS, ScannerCategory.MODEL]
    if mode == "all":
        return ExecutionPlan(categories=categories, continue_on_fail=True)

    return ExecutionPlan(categories=categories, continue_on_fail=False, model_requires_previous_pass=True)


@app.command()
def scan(
    dataset: str = typer.Option("data/train.parquet", help="Путь к датасету (внутри data/)"),
    model: str = typer.Option("models/model.pt", help="Путь к модели (внутри models/)"),
    mode: str = typer.Option(
        "default",
        help="default: sanity+stats, model только если прошло; all: запустить все; only: одна категория",
        case_sensitive=False,
    ),
    only: str = typer.Option(
        "sanity",
        help="Категория для режима only: sanity, stats, model",
        case_sensitive=False,
    ),
    continue_on_fail: bool = typer.Option(
        False,
        help="Продолжать выполнение при ошибках (актуально для only/all)",
    ),
    reference: str = typer.Option(
        None,
        help="Режим 2: опорная модель (NLP — папка/safetensors; табличный с --target — CSV чистой версии)",
    ),
    clean_data: str = typer.Option(
        None,
        help="Режим 2: чистый сэмпл -> калибровка (NLP). В табличном режиме берётся из --reference",
    ),
    target: str = typer.Option(
        None,
        help="Табличный режим: колонка-таргет -> обучить MLP (общая стандартизация) в одном scan",
    ),
    drop: str = typer.Option(
        "",
        help="Табличный режим: служебные колонки через запятую",
    ),
):
    ref_model, clean = reference, clean_data
    if target:  # табличный режим: обучить модель(и) с ОБЩЕЙ стандартизацией -> обычный пайплайн
        from pathlib import Path as _P
        from tabular_prep import prepare
        nm = f"{_P(dataset).parent.name}_{_P(dataset).stem}".strip("_") or "case"
        prepared = prepare(dataset, target,
                           [d.strip() for d in drop.split(",") if d.strip()],
                           clean_csv=reference, name=nm)
        dataset, model = prepared["dataset"], prepared["model"]
        ref_model, clean = prepared["reference_model"], prepared["clean_data"]
        print("[tabular] обучен MLP (общая стандартизация)"
              + (" + опорная модель + калибровка" if ref_model else " (режим 1, триаж)"))

    discover_scanners()
    plan = _build_plan(mode.lower(), only.lower(), continue_on_fail)
    scanners = registry.build(plan.categories)
    # Прокидываем опору режима 2 в адаптер post-train (без изменения ядра/ScanContext).
    for s in scanners:
        if hasattr(s, "reference_model_path"):
            s.reference_model_path = ref_model
        if hasattr(s, "clean_data_path"):
            s.clean_data_path = clean
    pipeline = SecurityPipeline(scanners)

    report = pipeline.execute(dataset, model, plan)

    if report.overall_status.value != "passed":
        raise typer.Exit(code=1)


@app.command()
def init_demo() -> None:
    """Create demo dataset and model under data/ and models/."""
    dataset_path, model_path = init_demo_assets()
    print(f"Demo dataset created: {dataset_path}")
    print(f"Demo model created: {model_path}")


if __name__ == "__main__":
    app()
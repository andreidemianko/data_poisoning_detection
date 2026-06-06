"""
Готовит кейсовый датасет для model-сканера проекта:
  data/<name>.csv   — признаки (числовые, стандартизованные) + колонка label
  models/<name>.pt  — обученный на них MLP (2 скрытых слоя), torch state_dict

Зачем так:
  * Сканер ищет метку в колонках label/target/y/class — поэтому таргет
    переименовывается в `label`.
  * Признаки стандартизуются и сохраняются уже в таком виде — это и есть
    представление, которое видит и модель, и сканер (так их вход совпадает;
    у банковских признаков очень разные масштабы, без этого MLP учится плохо).
  * Модель — это нейросеть (MLP), потому что model-сканер реконструирует forward
    из state_dict. Базовые XGBoost-модели кейса сканер читать не умеет (у деревьев
    нет активаций) — здесь обучается репрезентативный MLP, т.е. та модель, которую
    и инспектировал бы model-слой гейта.

Пример (taiwan, отравленный dataset_3):
  python3 prep_case_dataset.py \
      --src /path/to/security_case_datasets/taiwan_credit_default/dataset_3.csv \
      --target "default.payment.next.month" --name taiwan_d3
  python3 -m src.cli scan --dataset data/taiwan_d3.csv --model models/taiwan_d3.pt --mode only --only model
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def find_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "requirements.txt").exists() or (cand / "src" / "core").is_dir():
            return cand
    return start


def main() -> None:
    ap = argparse.ArgumentParser(description="case dataset -> project-ready data/ + models/")
    ap.add_argument("--src", required=True, help="путь к CSV/parquet кейс-датасета")
    ap.add_argument("--target", required=True, help="имя колонки-таргета в исходнике")
    ap.add_argument("--name", required=True, help="короткое имя выходных файлов (напр. taiwan_d3)")
    ap.add_argument("--drop", nargs="*", default=[], help="лишние колонки (id и т.п.), через пробел")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    data_dir = root / "data"
    model_dir = root / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.src)
    df = pd.read_parquet(src) if src.suffix.lower() == ".parquet" else pd.read_csv(src)
    if args.target not in df.columns:
        raise SystemExit(f"колонка-таргет {args.target!r} не найдена; есть: {list(df.columns)}")

    y = pd.factorize(df[args.target], sort=True)[0].astype("int64")
    feat = df.drop(columns=[args.target, *args.drop], errors="ignore").select_dtypes(include="number")
    if feat.shape[1] == 0:
        raise SystemExit("после отбрасывания таргета/--drop не осталось числовых признаков")

    X = feat.to_numpy(dtype="float32")
    # медианная импутация + стандартизация (вход модели И сканера)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1e-9
    X = (X - mu) / sd

    # сохраняем то, что пойдёт в гейт: стандартизованные признаки + label
    out = pd.DataFrame(X, columns=list(feat.columns))
    out["label"] = y
    out_csv = data_dir / f"{args.name}.csv"
    out.to_csv(out_csv, index=False)

    # обучаем MLP (2 скрытых слоя, n_classes выходов)
    n_cls = int(y.max()) + 1
    d = X.shape[1]
    torch.manual_seed(0)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    model = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_cls),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    out_pt = model_dir / f"{args.name}.pt"
    torch.save(model.state_dict(), out_pt)

    model.eval()
    with torch.no_grad():
        acc = float((model(Xt).argmax(1).numpy() == y).mean())
    rel_csv = out_csv.relative_to(root)
    rel_pt = out_pt.relative_to(root)
    print(f"created {rel_csv}  ({len(out)} строк, {d} признаков, {n_cls} класса)")
    print(f"created {rel_pt}   (MLP {d}-64-32-{n_cls}, train acc={acc:.3f})")
    print("\nтеперь запусти model-сканер:")
    print(f"  python3 -m src.cli scan --dataset {rel_csv} --model {rel_pt} --mode only --only model")


if __name__ == "__main__":
    main()

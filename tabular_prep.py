"""
Авто-подготовка табличного кейса для ЕДИНОГО `python3 -m src.cli scan`.

Когда сканеру дают сырой табличный датасет (--target задан), CLI зовёт prepare():
обучает модель-кандидата (и опорную, если дан чистый эталон) с ОБЩЕЙ
стандартизацией — scaler фитится на чистом эталоне и применяется к обоим, поэтому
кандидат/опора/чистый-сэмпл в одной шкале (иначе калибровка и model_diff «плывут»).

Пишет во временный gitignored-кэш под data/.cli_cache и models/.cli_cache и
возвращает пути, которые скармливаются ОБЫЧНОМУ пайплайну (ядро не меняется:
модели лежат под models/, как требует core.loaders).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CACHE_D = ROOT / "data" / ".cli_cache"
CACHE_M = ROOT / "models" / ".cli_cache"


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _read(p):
    p = _resolve(p); s = p.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(p)
    if s == ".jsonl":
        return pd.read_json(p, lines=True)
    return pd.read_csv(p)


def _xy(df, target, drop):
    y = pd.factorize(df[target], sort=True)[0].astype("int64")
    feat = df.drop(columns=[target, *drop], errors="ignore").select_dtypes(include="number")
    if feat.shape[1] == 0:
        raise SystemExit("после отбрасывания таргета/--drop не осталось числовых признаков")
    X = feat.to_numpy("float32")
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0); med = np.where(np.isnan(med), 0.0, med)
        ii = np.where(np.isnan(X)); X[ii] = np.take(med, ii[1])
    return X, y, list(feat.columns)


def _train(X, y, epochs):
    torch.manual_seed(0)
    n_cls = int(y.max()) + 1
    m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                      nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_cls))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y); m.train()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
    return m


def prepare(candidate_csv, target, drop=(), clean_csv=None, name="case", epochs=300):
    """Возвращает project-relative пути dataset/model (для core-пайплайна) и
    абсолютные reference_model/clean_data (для адаптера). reference_model и
    clean_data заполняются, только если дан clean_csv (чистый эталон)."""
    CACHE_D.mkdir(parents=True, exist_ok=True); CACHE_M.mkdir(parents=True, exist_ok=True)
    drop = list(drop)
    cand = _read(candidate_csv)
    if target not in cand.columns:
        raise SystemExit(f"таргет {target!r} не найден; есть: {list(cand.columns)}")

    # ОБЩАЯ стандартизация: mu/sd по чистому эталону (если есть), иначе по кандидату
    if clean_csv:
        clean = _read(clean_csv)
        Xc, yc, cols = _xy(clean, target, drop)
        mu, sd = Xc.mean(0), Xc.std(0)
    Xk, yk, colsk = _xy(cand, target, drop)
    if not clean_csv:
        mu, sd, cols = Xk.mean(0), Xk.std(0), colsk
    sd[sd < 1e-9] = 1e-9

    Xk_s = (Xk - mu) / sd
    cdf = pd.DataFrame(Xk_s, columns=colsk); cdf["label"] = yk
    (CACHE_D / f"{name}_cand.csv").write_text(cdf.to_csv(index=False))
    torch.save(_train(Xk_s, yk, epochs).state_dict(), CACHE_M / f"{name}_cand.pt")
    out = {"dataset": f"data/.cli_cache/{name}_cand.csv",
           "model": f"models/.cli_cache/{name}_cand.pt",
           "reference_model": None, "clean_data": None}

    if clean_csv:
        Xc_s = (Xc - mu) / sd
        cl = pd.DataFrame(Xc_s, columns=cols); cl["label"] = yc
        (CACHE_D / f"{name}_clean.csv").write_text(cl.to_csv(index=False))
        torch.save(_train(Xc_s, yc, epochs).state_dict(), CACHE_M / f"{name}_ref.pt")
        out["reference_model"] = str(CACHE_M / f"{name}_ref.pt")
        out["clean_data"] = str(CACHE_D / f"{name}_clean.csv")
    return out
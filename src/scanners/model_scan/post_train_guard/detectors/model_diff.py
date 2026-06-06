"""
============================================================================
 model_diff — поведенческое сравнение НОВОЙ модели с ОПОРНОЙ (старой/принятой)
============================================================================

Это НЕ обёртка над Spectral/AC/RPP. Отдельный детектор: прогоняет ДВЕ модели
(опорную и кандидата) на одних данных и сравнивает их ВЫХОДНЫЕ вероятности
построчно (Total Variation). Backdoor/label-flip строка: новая модель ведёт
себя иначе старой → большое расхождение; чистая строка → модели согласны (TV≈0).

ПОЧЕМУ ЭТО ДАЁТ АБСОЛЮТНЫЙ ВЕРДИКТ (в отличие от одиночных детекторов):
опорная модель — это точка отсчёта («шкала»). Расхождение самокалибруется
вокруг нуля (согласия), поэтому порог осмыслен: TV>0.5 = модели кладут
большинство массы на РАЗНЫЕ классы. Единственный детектор, который может
честно вернуть PASSED (нет дрейфа → ALLOW) и BLOCK (целевой кластер → decline).

ВХОД: опорная модель (reference_model_path / reference_model_state) + кандидат
(model_path для BERT / model_state для табличного MLP) + данные. Если опоры нет —
детектор СКИПается (режим 1, триаж остаётся на Spectral/AC/RPP).

ОГОВОРКИ (для защиты):
  * слеп к закладке, ОБЩЕЙ со старой моделью (обе согласны → TV≈0);
  * естественный сдвиг данных между версиями тоже даёт расхождение → отделяется
    тем, что backdoor = КОНЦЕНТРИРОВАННЫЙ кластер сильного расхождения, а дрейф —
    размазанное умеренное; широкое расхождение (>broad_frac) трактуется как
    дрейф → REVIEW, не BLOCK;
  * сравнение по предсказаниям предполагает согласованное пространство меток
    (индекс класса означает одно и то же у обеих моделей).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import common as cm
from . import model_level as ml
from . import nlp_model_level as nlp
from ..models import Finding, FindingStatus

_NAME, _CAT = "Model-diff vs reference model", "model-diff"


# --------------------------------------------------------------------------- #
#  predict_proba для обеих модальностей
# --------------------------------------------------------------------------- #
def _proba_nlp(model_path: str, df, max_len: int = 128, batch: int = 64) -> np.ndarray:
    import torch
    if not nlp.is_transformer_dir(nlp.resolve_model_dir(model_path)):
        raise ValueError("reference/candidate is not a HuggingFace dir (no config.json)")
    tok, mdl, dev = nlp.load_text_classifier(nlp.resolve_model_dir(model_path))
    text_cols, _, reason = nlp.find_text_label(df)
    if not text_cols:
        raise ValueError(reason or "no text columns")
    texts = nlp._texts_of(df, text_cols)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            out.append(torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def _proba_tabular_pair(old_state, new_state, df) -> Tuple[np.ndarray, np.ndarray]:
    layers_new, r1 = ml.reconstruct_linear_layers(new_state)
    if layers_new is None:
        raise ValueError(f"candidate model: {r1}")
    layers_old, r2 = ml.reconstruct_linear_layers(old_state)
    if layers_old is None:
        raise ValueError(f"reference model: {r2}")
    X, _, _ = cm.extract_xy(df)
    for layers, who in ((layers_new, "candidate"), (layers_old, "reference")):
        if X.shape[1] != int(layers[0][0].shape[1]):
            raise ValueError(f"{who} feature width {int(layers[0][0].shape[1])} != data {X.shape[1]}")
    _, p_new = ml.mlp_forward(layers_new, X)
    _, p_old = ml.mlp_forward(layers_old, X)
    return p_old, p_new


def tv_distance(p_old: np.ndarray, p_new: np.ndarray) -> np.ndarray:
    """Total Variation построчно. Разное число классов → дополняем нулями до
    объединённого носителя (новый/исчезнувший класс)."""
    k = max(p_old.shape[1], p_new.shape[1])

    def pad(p):
        if p.shape[1] == k:
            return p
        z = np.zeros((p.shape[0], k), dtype=float)
        z[:, :p.shape[1]] = p
        return z
    return 0.5 * np.abs(pad(p_old) - pad(p_new)).sum(axis=1)


# --------------------------------------------------------------------------- #
#  Решение из распределения TV (accept / review / decline)
# --------------------------------------------------------------------------- #
def decide_from_tv(tv: np.ndarray, cfg) -> Tuple[FindingStatus, str, dict]:
    tv_thr = float(cfg.get("model_diff_tv", 0.5))
    broad = float(cfg.get("model_diff_broad_frac", 0.40))
    strong = float(cfg.get("model_diff_strong_tv", 0.70))
    flagged = tv >= tv_thr
    frac = float(flagged.mean())
    det = {
        "flagged": int(flagged.sum()),
        "flagged_fraction": round(frac, 4),
        "tv_threshold": tv_thr,
        "median_tv_all": round(float(np.median(tv)), 4),
        "median_tv_flagged": round(float(np.median(tv[flagged])), 4) if flagged.any() else 0.0,
        "top_divergent_rows": np.argsort(tv)[::-1][:min(20, len(tv))].astype(int).tolist(),
    }
    if frac == 0.0:
        return FindingStatus.PASSED, "no behavioral drift vs reference model (ALLOW)", det
    cluster_med = float(np.median(tv[flagged]))
    if 0.0 < frac <= broad and cluster_med >= strong:
        return (FindingStatus.BLOCK,
                f"targeted divergent cluster vs reference: {int(flagged.sum())} rows "
                f"(median TV={cluster_med:.2f}) — likely backdoor/poison (DECLINE)", det)
    return (FindingStatus.REVIEW,
            f"behavioral divergence vs reference on {int(flagged.sum())} rows "
            f"(median TV={cluster_med:.2f}) — review: targeted poison vs legitimate drift", det)


# --------------------------------------------------------------------------- #
#  Registry check (СКИП, если опоры нет)
# --------------------------------------------------------------------------- #
def check_model_diff(inp) -> Finding:
    old_ref = inp.reference_model_state if getattr(inp, "reference_model_state", None) is not None \
        else getattr(inp, "reference_model_path", None)
    if old_ref is None:
        return Finding(_NAME, _CAT, FindingStatus.SKIPPED,
                       "no reference model provided (Mode 1: rank-triage only)")
    new_is_nlp = bool(inp.model_path) and nlp.is_transformer_dir(nlp.resolve_model_dir(inp.model_path))
    try:
        if new_is_nlp:
            if not isinstance(old_ref, str):
                return Finding(_NAME, _CAT, FindingStatus.SKIPPED,
                               "reference is a state_dict but candidate is a transformer — modality mismatch")
            p_old, p_new = _proba_nlp(old_ref, inp.dataset), _proba_nlp(inp.model_path, inp.dataset)
        else:
            if inp.model_state is None:
                return Finding(_NAME, _CAT, FindingStatus.SKIPPED, "no candidate model_state (tabular)")
            old_state = old_ref if isinstance(old_ref, dict) else _load_state(old_ref)
            p_old, p_new = _proba_tabular_pair(old_state, inp.model_state, inp.dataset)
    except ImportError:
        return Finding(_NAME, _CAT, FindingStatus.SKIPPED, "transformers/torch not installed")
    except ValueError as exc:
        return Finding(_NAME, _CAT, FindingStatus.SKIPPED, str(exc))
    except Exception as exc:  # noqa: BLE001
        return Finding(_NAME, _CAT, FindingStatus.ERROR, "model_diff failed",
                       {"error_type": type(exc).__name__, "error": str(exc)})
    if len(p_old) != len(p_new):
        return Finding(_NAME, _CAT, FindingStatus.SKIPPED, "reference/candidate row count mismatch")
    tv = tv_distance(p_old, p_new)
    if len(tv) < inp.config.get("min_rows", 20):
        return Finding(_NAME, _CAT, FindingStatus.SKIPPED, f"too few rows ({len(tv)})")
    status, verdict, det = decide_from_tv(tv, inp.config)
    det["n_rows"] = int(len(tv))
    return Finding(_NAME, _CAT, status, verdict, det)


def _load_state(path):
    from ..loaders import load_state_dict
    return load_state_dict(path)


# --------------------------------------------------------------------------- #
#  Самотест (proba-уровень, без torch/моделей): TV + решение
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n, K = 1000, 3
    base = rng.dirichlet(np.ones(K) * 0.3, size=n)          # «уверенные» предсказания
    cfg = {"model_diff_tv": 0.5, "model_diff_broad_frac": 0.40, "model_diff_strong_tv": 0.70}

    # 1) backdoor: 20% строк новая модель жёстко флипает в др. класс, старая нет
    p_old, p_new = base.copy(), base.copy()
    pois = np.zeros(n, bool); pois[:200] = True
    p_old[pois] = np.array([0.025, 0.95, 0.025])            # старая: класс 1
    p_new[pois] = np.array([0.95, 0.025, 0.025])            # новая: класс 0 (цель)
    s, v, d = decide_from_tv(tv_distance(p_old, p_new), cfg)
    print(f"[backdoor]  {s.value:7s} | {v}")
    print(f"            flagged={d['flagged']} frac={d['flagged_fraction']} med_TV_flagged={d['median_tv_flagged']}")

    # 2) идентичные модели (нет дрейфа) -> ALLOW
    s2, v2, _ = decide_from_tv(tv_distance(base, base), cfg)
    print(f"[identical] {s2.value:7s} | {v2}")

    # 3) размазанный лёгкий дрейф (не кластер) -> REVIEW, не BLOCK
    p3 = base.copy(); idx = rng.choice(n, 150, replace=False)
    p3[idx] = 0.6 * p3[idx] + 0.4 * rng.dirichlet(np.ones(K), size=150)
    s3, v3, d3 = decide_from_tv(tv_distance(base, p3), cfg)
    print(f"[drift]     {s3.value:7s} | {v3}  (frac={d3['flagged_fraction']})")

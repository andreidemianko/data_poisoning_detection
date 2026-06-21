"""
Active registry of the POST-TRAIN ensemble: three model-level methods —
SPECTRE (robust Spectral Signatures + QUE), Activation Clustering, RPP — each applied to tabular models
(reconstructed MLP) and to fine-tuned BERT (embeddings / input-embedding noise).

This is the "ensemble of three" the gate runs. The data-level / consensus
detectors (charset, trigger, SecureLearn, kNN) are parked in the top-level
`reserved/` package and are intentionally NOT part of this ensemble yet.

Add a post-train detector  -> write a check + add it to CHECKS.
Remove one                 -> delete it from CHECKS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from post_train_guard.detectors import common as cm
from post_train_guard.detectors import model_diff as md
from post_train_guard.detectors import model_level as ml
from post_train_guard.detectors import nlp_model_level as nlp
from post_train_guard.detectors import spectre as spct
from post_train_guard.models import Finding, FindingStatus

DEFAULT_CONFIG: Dict[str, float] = {
    "expected_poison": 0.15,    # доля верхних скоров для ранг-гейта (from_scores; теперь только reserved/)
    "review_frac": 0.02,        # (устар.) порог доли для старого MAD-гейта; не используется
    "sil_review": 0.45,         # (устар.) фикс-порог silhouette AC; режим 1 заменён на permutation-гейт
    "min_rows": 20,
    # --- режим 1 (без опоры): самокалиброванные гейты вместо «всегда top-ε» ---
    "perm_K": 12,               # перестановок меток для нуля (AC silhouette / SPECTRE gap)
    "perm_margin_sil": 0.05,    # запас над null95 для AC silhouette
    "sil_floor": 0.5,           # абс. нижний порог silhouette для флага AC (срез шума)
    "perm_margin_gap": 0.30,    # запас над null95 для SPECTRE (advisory: ранжирует, не голосует)
    "rpp_gap_sep": 1.2,         # разделение кластера стабильности (within-std) для RPP-гейта
    "nlp_rpp_perturb": 8,       # число прогонов BERT для RPP (меньше = быстрее)
    # --- model_diff (режим 2, есть опорная модель): порог абсолютного вердикта ---
    "model_diff_tv": 0.50,        # per-row TV: модели кладут массу на разные классы
    "model_diff_broad_frac": 0.40,  # >этой доли расхождения = широкий дрейф -> REVIEW, не BLOCK
    "model_diff_strong_tv": 0.70,   # медиана TV кластера для BLOCK (целевая закладка)
    # --- калибровка по чистому сэмплу (режим 2, есть старые данные) ---
    "calib_alpha": 0.01,          # квантиль чистых скоров для порога (1-alpha); фон превышения
    "calib_bootstrap": 200,       # число бутстрап-итераций для верхней CI порога
    "calib_review_mult": 3.0,     # доля кандидата выше порога > alpha*mult -> REVIEW
    "calib_block_frac": 0.10,     # доля кандидата выше порога >= этого -> BLOCK (decline)
}


@dataclass
class ScanInput:
    """Единый вход для всех проверок."""
    dataset: Any                                   # pandas.DataFrame
    model_state: Optional[Dict[str, Any]] = None   # state_dict обычной модели (или None)
    model_path: Optional[str] = None               # путь к модели/папке BERT (или None)
    config: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    # --- ОПОРА для режима 2 (необязательная): старая/принятая модель ---
    reference_model_path: Optional[str] = None     # путь к опорной BERT-папке (или None)
    reference_model_state: Optional[Dict[str, Any]] = None  # state_dict опорного MLP (или None)
    # --- ОПОРА для режима 2 (необязательная): чистый сэмпл данных для калибровки ---
    clean_data: Any = None                         # pandas.DataFrame доверенных строк (или None)


# Фабрики Finding (используются и активными, и зарезервированными проверками)
def skip(name, cat, reason):
    return Finding(name, cat, FindingStatus.SKIPPED, reason)


def err(name, cat, reason, exc):
    return Finding(name, cat, FindingStatus.ERROR, reason,
                   {"error_type": type(exc).__name__, "error": str(exc)})


def from_scores(name, cat, scores, cfg, verdict_review, verdict_ok, extra=None):
    """Самодостаточный РАНГ-гейт (без чистого эталона): помечаем верхние
    expected_poison строк по скору как кандидатов на просмотр. Это ТРИАЖ, а не
    калиброванный вердикт — без эталона model-level не отличает чистое от
    отравленного (форма скоров и консенсус детекторов не разделяют их). Поэтому
    статус всегда REVIEW, а ценность — РАНЖИРОВАНИЕ (top_suspicious_rows).
    `verdict_ok` больше не используется (PASS без эталона недостижим)."""
    eps = float(cfg.get("expected_poison", 0.15))
    fl = cm.top_fraction_flags(scores, eps)
    det = {"flagged": int(fl.sum()), "flagged_fraction": round(float(fl.mean()), 4),
           "gate": f"rank top-{eps:.0%} (self-contained, no clean ref)",
           "top_suspicious_rows": cm.top_indices(scores), **(extra or {})}
    return Finding(name, cat, FindingStatus.REVIEW,
                   f"top-{eps:.0%} suspect rows surfaced for review "
                   f"({int(fl.sum())} {verdict_review})", det, advisory=True)


def from_scores_calibrated(name, cat, clean_scores, cand_scores, cfg, verdict_review, extra=None):
    """КАЛИБРОВАННЫЙ гейт (есть чистый сэмпл): порог = верхняя бутстрап-CI
    (1-alpha)-квантили скоров ЧИСТЫХ строк; кандидатные строки выше порога — флаг.
    На чистом кандидате доля превышения ≈ alpha (фон); значимо выше → аномалия.
    Это ВЕРДИКТ (advisory=False) — ведёт решение по политике from_findings."""
    alpha = float(cfg.get("calib_alpha", 0.01))
    tau = cm.bootstrap_upper_quantile(np.asarray(clean_scores), 1.0 - alpha,
                                      int(cfg.get("calib_bootstrap", 200)))
    cand = np.asarray(cand_scores, dtype=float)
    flagged = cand > tau
    frac = float(flagged.mean()) if len(cand) else 0.0
    det = {"flagged": int(flagged.sum()), "flagged_fraction": round(frac, 4),
           "calib_tau": round(float(tau), 6), "clean_baseline_alpha": alpha,
           "n_clean": int(len(clean_scores)),
           "gate": "calibrated vs clean sample (bootstrap upper-CI threshold)",
           "top_suspicious_rows": cm.top_indices(cand), **(extra or {})}
    if frac >= float(cfg.get("calib_block_frac", 0.10)):
        return Finding(name, cat, FindingStatus.BLOCK,
                       f"{int(flagged.sum())} {verdict_review} far beyond clean baseline (DECLINE)", det)
    if frac > alpha * float(cfg.get("calib_review_mult", 3.0)):
        return Finding(name, cat, FindingStatus.REVIEW,
                       f"{int(flagged.sum())} {verdict_review} above clean sample (review)", det)
    return Finding(name, cat, FindingStatus.PASSED,
                   "consistent with clean sample (calibrated ALLOW)", det)


def _tab_clean_reps(inp, layers, n_features):
    """reps чистого сэмпла через ТУ ЖЕ модель-кандидата (для калибровки)."""
    Xc, yc, _ = cm.extract_xy(inp.clean_data)
    if Xc.shape[1] != n_features:
        raise ValueError(f"clean sample feature width {Xc.shape[1]} != model {n_features}")
    reps_c, _ = ml.mlp_forward(layers, Xc)
    return reps_c, yc, Xc


# ---- РЕЖИМ 1 (без опоры): самокалиброванные гейты, AC — приоритетный ---------- #
# Вместо «всегда top-ε» (ранг-гейт) каждый метод сам решает PASS/REVIEW; чистое
# (без выраженных аномалий) проходит. Свод (from_findings) приоритезирует AC.
def _gap_stat(reps, labels):
    s = spct.spectre_scores(reps, labels)
    return float(np.percentile(s, 95) / (np.percentile(s, 50) + 1e-9))


def _sil_stat(reps, labels):
    return float(ml.activation_clustering(reps, labels)[2].get("max_minority_silhouette", -1.0))


def _spectre_mode1(name, cat, reps, y, cfg, extra):
    s = spct.spectre_scores(reps, y)
    fl, real, thr = cm.permutation_flag(_gap_stat, reps, y, K=int(cfg.get("perm_K", 12)),
                                        margin=float(cfg.get("perm_margin_gap", 0.3)))
    det = {"gate": "self-calibrated (permutation null)", "spectral_gap": real, "gap_thresh": thr,
           "top_suspicious_rows": cm.top_indices(s), **(extra or {})}
    st = FindingStatus.REVIEW if fl else FindingStatus.PASSED
    msg = f"SPECTRE anomaly: gap {real} > {thr}" if fl else f"no SPECTRE anomaly: gap {real} <= {thr}"
    # role="advisory": сильное РАНЖИРОВАНИЕ строк, но self-гейт шумит на чистом
    # (per-class whitening ловит ПРИРОДНУЮ структуру классов) -> не голосует, только строки.
    return Finding(name, cat, st, msg, det, advisory=True, role="advisory")


def _ac_mode1(name, cat, reps, y, cfg, extra):
    _flags, scores, meta = ml.activation_clustering(reps, y)
    sil = meta.get("max_minority_silhouette", -1.0)
    fl, real, thr = cm.permutation_flag(_sil_stat, reps, y, K=int(cfg.get("perm_K", 12)),
                                        margin=float(cfg.get("perm_margin_sil", 0.05)),
                                        floor=float(cfg.get("sil_floor", 0.5)))
    det = {"gate": "self-calibrated (permutation null)", "max_minority_silhouette": sil,
           "sil_thresh": thr, "top_suspicious_rows": cm.top_indices(scores), **(extra or {})}
    st = FindingStatus.REVIEW if fl else FindingStatus.PASSED
    msg = f"tight minority cluster: sil {real} > {thr}" if fl else f"no significant cluster: sil {real} <= {thr}"
    return Finding(name, cat, st, msg, det, advisory=True, role="primary")   # AC — приоритетный


def _rpp_mode1(name, cat, scores, cfg, extra):
    fl, frac, sep = cm.cluster_gap_flag(scores, sep=float(cfg.get("rpp_gap_sep", 1.2)))
    det = {"gate": "self-contained stability gap", "hi_cluster_frac": frac, "separation": sep,
           "top_suspicious_rows": cm.top_indices(scores), **(extra or {})}
    st = FindingStatus.REVIEW if fl else FindingStatus.PASSED
    msg = (f"separated stable subpopulation (frac {frac}, sep {sep})" if fl
           else "no separated stable subpopulation")
    # role="corroborator": self-гейт чистый на эталоне (0 ложных) -> голосует наравне с AC.
    return Finding(name, cat, st, msg, det, advisory=True, role="corroborator")


# ---- Post-train, tabular (reconstructs an MLP from the state_dict) ---------- #
def check_spectral(inp: ScanInput) -> Finding:
    n, c = "SPECTRE (model, robust whitening + QUE)", "tabular-model"
    try:
        layers, X, y, label_col = ml.prepare_model(inp.model_state, inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    except Exception as exc:  # noqa: BLE001
        return err(n, c, "prepare failed", exc)
    if len(X) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(X)})")
    repr_, _ = ml.mlp_forward(layers, X)
    if inp.clean_data is not None:                       # режим 2: калибровка по чистому сэмплу
        try:
            reps_c, yc, _ = _tab_clean_reps(inp, layers, X.shape[1])
            comb = np.vstack([reps_c, repr_]); yy = np.concatenate([yc, y])
            sa = spct.spectre_scores(comb, yy)
            return from_scores_calibrated(n, c, sa[:len(reps_c)], sa[len(reps_c):], inp.config,
                                          "SPECTRE-outlier rows",
                                          {"label_column": label_col, "n_rows": len(X), "n_clean": len(reps_c)})
        except ValueError:
            pass                                        # калибровка неприменима -> триаж ниже
    return _spectre_mode1(n, c, repr_, y, inp.config, {"label_column": label_col, "n_rows": len(X)})


def check_activation_clustering(inp: ScanInput) -> Finding:
    n, c = "Activation Clustering (model)", "tabular-model"
    try:
        layers, X, y, label_col = ml.prepare_model(inp.model_state, inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    except Exception as exc:  # noqa: BLE001
        return err(n, c, "prepare failed", exc)
    if len(X) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(X)})")
    repr_, _ = ml.mlp_forward(layers, X)
    if inp.clean_data is not None:                       # режим 2: калибровка по чистому сэмплу
        try:
            reps_c, yc, _ = _tab_clean_reps(inp, layers, X.shape[1])
            comb = np.vstack([reps_c, repr_]); yy = np.concatenate([yc, y])
            _f, scores_all, meta = ml.activation_clustering(comb, yy)
            return from_scores_calibrated(n, c, scores_all[:len(reps_c)], scores_all[len(reps_c):],
                                          inp.config, "rows far from clean clusters",
                                          {"label_column": label_col, "n_rows": len(X),
                                           "n_clean": len(reps_c), **meta})
        except ValueError:
            pass
    return _ac_mode1(n, c, repr_, y, inp.config, {"label_column": label_col, "n_rows": len(X)})


def check_rpp(inp: ScanInput) -> Finding:
    n, c = "RPP (model, prediction stability)", "tabular-model"
    try:
        layers, X, y, label_col = ml.prepare_model(inp.model_state, inp.dataset)
    except ValueError as exc:
        return skip(n, c, str(exc))
    except Exception as exc:  # noqa: BLE001
        return err(n, c, "prepare failed", exc)
    if len(X) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(X)})")
    if inp.clean_data is not None:                       # режим 2: калибровка (RPP model-anchored)
        try:
            Xc, _yc, _ = cm.extract_xy(inp.clean_data)
            if Xc.shape[1] != X.shape[1]:
                raise ValueError("clean sample feature width mismatch")
            cs, ds = ml.rpp_scores(layers, Xc), ml.rpp_scores(layers, X)
            return from_scores_calibrated(n, c, cs, ds, inp.config, "abnormally-stable rows",
                                          {"label_column": label_col, "n_rows": len(X), "n_clean": len(Xc)})
        except ValueError:
            pass
    s = ml.rpp_scores(layers, X)
    return _rpp_mode1(n, c, s, inp.config, {"label_column": label_col, "n_rows": len(X)})


# ---- Post-train, NLP (same three methods on the fine-tuned BERT) ------------ #
def _nlp_repr_check(inp, name, fn_on_repr):
    c = "nlp-model"
    try:
        emb, y, label_col, tcols = nlp.representation(inp.model_path, inp.dataset)
    except ValueError as exc:
        return skip(name, c, str(exc))
    except ImportError:
        return skip(name, c, "transformers/torch not installed: pip install transformers")
    except Exception as exc:  # noqa: BLE001
        return err(name, c, "BERT encode failed", exc)
    if len(emb) < inp.config["min_rows"]:
        return skip(name, c, f"too few rows ({len(emb)})")
    return fn_on_repr(emb, y, label_col, tcols)


def check_nlp_spectral(inp: ScanInput) -> Finding:
    n = "SPECTRE on BERT embeddings"

    def on_repr(emb, y, label_col, tcols):
        if inp.clean_data is not None:                  # режим 2: калибровка по чистому сэмплу
            try:
                emb_c, yc, _, _ = nlp.representation(inp.model_path, inp.clean_data)
                comb = np.vstack([emb_c, emb]); yy = np.concatenate([yc, y])
                sa = spct.spectre_scores(comb, yy)
                return from_scores_calibrated(n, "nlp-model", sa[:len(emb_c)], sa[len(emb_c):], inp.config,
                                              "SPECTRE-outlier rows in BERT space",
                                              {"text_columns": tcols, "label_column": label_col,
                                               "n_rows": len(emb), "n_clean": len(emb_c)})
            except ValueError:
                pass
        return _spectre_mode1(n, "nlp-model", emb, y, inp.config,
                              {"text_columns": tcols, "label_column": label_col, "n_rows": len(emb)})
    return _nlp_repr_check(inp, n, on_repr)


def check_nlp_activation_clustering(inp: ScanInput) -> Finding:
    n = "Activation Clustering on BERT embeddings"

    def on_repr(emb, y, label_col, tcols):
        if inp.clean_data is not None:                  # режим 2: калибровка по чистому сэмплу
            try:
                emb_c, yc, _, _ = nlp.representation(inp.model_path, inp.clean_data)
                comb = np.vstack([emb_c, emb]); yy = np.concatenate([yc, y])
                _f, scores_all, meta = ml.activation_clustering(comb, yy)
                return from_scores_calibrated(n, "nlp-model", scores_all[:len(emb_c)], scores_all[len(emb_c):],
                                              inp.config, "rows far from clean clusters in BERT space",
                                              {"text_columns": tcols, "label_column": label_col,
                                               "n_rows": len(emb), "n_clean": len(emb_c), **meta})
            except ValueError:
                pass
        return _ac_mode1(n, "nlp-model", emb, y, inp.config,
                         {"text_columns": tcols, "label_column": label_col, "n_rows": len(emb)})
    return _nlp_repr_check(inp, n, on_repr)


def check_nlp_rpp(inp: ScanInput) -> Finding:
    n, c = "RPP on BERT (input-embedding noise)", "nlp-model"
    try:
        s, label_col, tcols = nlp.rpp_scores_bert(
            inp.model_path, inp.dataset, n_perturb=int(inp.config["nlp_rpp_perturb"]))
    except ValueError as exc:
        return skip(n, c, str(exc))
    except ImportError:
        return skip(n, c, "transformers/torch not installed: pip install transformers")
    except Exception as exc:  # noqa: BLE001
        return err(n, c, "RPP(BERT) failed", exc)
    if len(s) < inp.config["min_rows"]:
        return skip(n, c, f"too few rows ({len(s)})")
    if inp.clean_data is not None:                       # режим 2: калибровка (RPP model-anchored)
        try:
            cs, _, _ = nlp.rpp_scores_bert(inp.model_path, inp.clean_data,
                                           n_perturb=int(inp.config["nlp_rpp_perturb"]))
            return from_scores_calibrated(n, c, cs, s, inp.config, "abnormally-stable rows (backdoor-like)",
                                          {"text_columns": tcols, "label_column": label_col,
                                           "n_rows": len(s), "n_clean": len(cs)})
        except (ValueError, ImportError):
            pass
    return _rpp_mode1(n, c, s, inp.config, {"text_columns": tcols, "label_column": label_col, "n_rows": len(s)})


# POST-TRAIN АНСАМБЛЬ: три метода x {табличный MLP, BERT} + model_diff.
# Добавить/убрать post-train детектор = правка этого списка.
# check_model_diff СКИПается, если опорная модель не передана (режим 1).
CHECKS: List[Callable[[ScanInput], Finding]] = [
    check_spectral,
    check_activation_clustering,
    check_rpp,
    check_nlp_spectral,
    check_nlp_activation_clustering,
    check_nlp_rpp,
    md.check_model_diff,
]

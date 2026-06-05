"""
============================================================================
 SPECTRE — Defending Against Backdoor Attacks Using Robust Statistics
 (Hayase, Kong, Somani, Oh — ICML 2021, arXiv:2104.11315)
============================================================================

ЗАЧЕМ: это принципиальный апгрейд Spectral Signatures (Tran et al. 2018).
Ванильный Spectral берёт топ-сингулярный вектор активаций и метит выбросы по
проекции на него. Он слепнет, когда отравление лежит НЕ вдоль направления
максимальной дисперсии чистых данных (его «съедает» естественная анизотропия
представления) — ровно тот false-positive/слепота-механизм, что мы замеряли.

SPECTRE чинит это двумя шагами:
  1. РОБАСТНОЕ ОТБЕЛИВАНИЕ (whitening). Оценить ЧИСТУЮ ковариацию (исключив
     подозреваемое отравление) и отбелить представления Σ^{-1/2}. После этого
     естественная анизотропия убрана, и отравление, образующее плотный сдвиг,
     торчит как выброс. (Если отбеливать ЭМПИРИЧЕСКОЙ ковариацией всех данных,
     отбелённая ковариация = I и сигнал исчезает — поэтому оценка обязана быть
     робастной, т.е. считаться по инлайерам.)
  2. QUE-СКОР (Quantum Entropy). Вместо проекции на один вектор — квадратичная
     форма с матричной экспонентой Q_α = exp(α (Σ̃−I)/(‖Σ̃‖₂−1)), которая
     усиливает спайк отбелённой ковариации. score_i = x̃ᵢᵀ Q_α x̃ᵢ / Tr(Q_α).
     При α→0 это сводится к ‖x̃ᵢ‖² (отбелённая норма); большие α фокусируются
     на топ-направлении Σ̃, где и концентрируется отрава.

ЧЕСТНЫЕ ОГОВОРКИ (для защиты):
  * Робастный оценщик ковариации из статьи (в духе Diakonikolas et al.)
    приближён ИТЕРАТИВНЫМ ОБРЕЗАНИЕМ (iterative trimming): несколько раундов
    «оценили Σ по инлайерам → отбелили → посчитали QUE → выкинули топ-1.5ε».
    Это даёт робастную Σ без полной SDP-машинерии оригинала.
  * Как и Spectral/AC, метод заточен под BACKDOOR (плотный кластер в
    представлении) и слеп к чистому label_flip.
  * Порог здесь — квантиль/робастный (MAD), conformal-калибровки нет.
  * Применяется ПО КЛАССАМ (как spectral_scores): целевой класс backdoor
    заранее неизвестен, поэтому скорим каждый класс отдельно.

Совместим со spectral_scores: spectre_scores(repr_, labels) -> per-row score
(выше = подозрительнее). Пока НЕ подключён в CHECKS — отдельный файл под
позднюю интеграцию.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Линейно-алгебраические хелперы (симметричные PSD-матрицы -> через eigh)
# --------------------------------------------------------------------------- #
def _inv_sqrt_psd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Σ^{-1/2} для симметричной PSD-матрицы через собственное разложение."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _que_scores(Xw: np.ndarray, alpha: float, spike_tol: float = 0.02) -> np.ndarray:
    """QUE-скор для отбелённых строк Xw (n,k), центрированных.

    Σ̃ = (1/n) Xwᵀ Xw; Q_α = exp(α (Σ̃−I)/(λ_max−1)); τᵢ = xᵢᵀ Q xᵢ / Tr(Q).
    Считаем через собственное разложение Σ̃ (без явного формирования Q):
        τᵢ = Σ_j q_j (xᵢ·v_j)² / Σ_j q_j,   q_j = exp(clip(α(λ_j−1)/(λ_max−1))).
    Если спайка нет (λ_max ≈ 1) — α→0 предел = отбелённая норма ‖x̃ᵢ‖²
    (иначе нормировка усилила бы чистый шум)."""
    n = len(Xw)
    Sigma = (Xw.T @ Xw) / max(n, 1)
    w, V = np.linalg.eigh(Sigma)                 # λ по возрастанию
    lam_max = float(w[-1])
    P = Xw @ V                                   # проекции на собств. базис (n,k)
    if lam_max - 1.0 < spike_tol:                # нет спайка -> отбелённая норма
        return (P * P).sum(axis=1)
    denom = lam_max - 1.0                         # нормировка: топ-экспонента = α
    expo = np.clip(alpha * (w - 1.0) / denom, -50.0, 50.0)
    q = np.exp(expo)                             # собственные значения Q
    return (P * P) @ q / q.sum()


# --------------------------------------------------------------------------- #
#  SPECTRE для ОДНОГО класса
# --------------------------------------------------------------------------- #
def spectre_class_scores(R: np.ndarray, k: int = 64, alpha: float = 4.0,
                         eps_poison: float = 0.10, n_iter: int = 4) -> np.ndarray:
    """QUE-скоры SPECTRE для активаций одного класса R (n,d).

    Шаги: PCA top-k -> РОБАСТНЫЙ старт инлайеров (ближайшие к покоординатной
    медиане, масштаб = MAD) -> итеративное отбеливание (Σ по инлайерам) ->
    QUE-скор по ВСЕМ отбелённым точкам. КРИТИЧНО: Σ оценивается по инлайерам
    (без отравы), иначе отбеливание эмпирической ковариацией даёт Σ̃=I и сигнал
    исчезает. Возвращает per-row score (выше = подозрительнее)."""
    n, d = R.shape
    k = int(min(k, d, max(1, n // 2)))
    if n < 10 or k < 2:
        return np.zeros(n)

    Rc = R - R.mean(axis=0, keepdims=True)
    # понижение размерности: топ-k правых сингулярных векторов
    try:
        _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(n)
    Xp = Rc @ Vt[:k].T                            # (n,k)

    n_keep = max(k + 1, int(round((1.0 - 1.5 * eps_poison) * n)))
    n_keep = min(n_keep, n)

    # РАУНД 1 — ДИАГОНАЛЬНОЕ робастное отбеливание (масштаб = MAD по координате).
    # Ключ к разрыву "курицы-яйца": MAD по координате игнорирует хвост отравы и
    # восстанавливает ЧИСТУЮ дисперсию оси, поэтому спайк отравы выживает в Σ̃
    # (эмпирическое отбеливание дало бы Σ̃=I и убило сигнал).
    med = np.median(Xp, axis=0)
    mad = np.median(np.abs(Xp - med), axis=0) * 1.4826
    mad[mad < 1e-9] = 1e-9
    Xw = (Xp - med) / mad
    tau = _que_scores(Xw, alpha)
    inlier = np.zeros(n, dtype=bool)
    inlier[np.argsort(tau)[:n_keep]] = True

    # РАУНДЫ 2+ — ПОЛНАЯ ковариация по инлайерам (декоррелирует; инлайеры уже
    # почти без отравы, значит Σ ≈ чистой). Скор — по ВСЕМ отбелённым точкам.
    for _ in range(max(0, n_iter - 1)):
        Xin = Xp[inlier]
        if len(Xin) <= k:
            break
        mu = Xin.mean(axis=0)
        cov = np.cov(Xin - mu, rowvar=False)
        Wsqrt = _inv_sqrt_psd(cov)
        Xw = (Xp - mu) @ Wsqrt
        tau = _que_scores(Xw, alpha)
        new = np.zeros(n, dtype=bool)
        new[np.argsort(tau)[:n_keep]] = True
        if np.array_equal(new, inlier):
            break
        inlier = new
    return tau


# --------------------------------------------------------------------------- #
#  SPECTRE по всем классам (drop-in аналог spectral_scores)
# --------------------------------------------------------------------------- #
def spectre_scores(repr_: np.ndarray, labels: np.ndarray, k: int = 64,
                   alpha: float = 4.0, eps_poison: float = 0.10,
                   n_iter: int = 4) -> np.ndarray:
    """Per-class SPECTRE QUE-скоры. Совместимо со spectral_scores(repr_, labels):
    выше score = подозрительнее. Класс <10 строк пропускается (score=0)."""
    scores = np.zeros(len(repr_))
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < 10:
            continue
        scores[idx] = spectre_class_scores(repr_[idx], k=k, alpha=alpha,
                                           eps_poison=eps_poison, n_iter=n_iter)
    return scores


def spectre_predict(repr_: np.ndarray, labels: np.ndarray,
                    expected_poison: float = 0.10, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Флаги по квантили (как у остальных бенч-детекторов) + непрерывный score
    (для ROC-AUC). Альтернатива — робастный MAD-порог на стороне gate."""
    s = spectre_scores(repr_, labels, eps_poison=expected_poison, **kw)
    thr = np.quantile(s, 1.0 - expected_poison)
    return (s >= thr).astype(int), s


# --------------------------------------------------------------------------- #
#  Самотест: синтетика, где ванильный Spectral слаб, а SPECTRE — силён
# --------------------------------------------------------------------------- #
def _vanilla_spectral(R: np.ndarray) -> np.ndarray:
    Rc = R - R.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    return (Rc @ Vt[0]) ** 2


def _auc(mask: np.ndarray, score: np.ndarray) -> float:
    # ROC-AUC без sklearn (ранговая форма Манна–Уитни)
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = mask.astype(bool); npos = pos.sum(); nneg = len(mask) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _mad_fp(scores, k=3.0):
    med = np.median(scores); mad = np.median(np.abs(scores - med)) * 1.4826
    return float((scores > med + k * mad).mean()) if mad > 1e-9 else 0.0


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    d, n_clean, n_pois = 64, 2000, 100

    def report(title, R, mask):
        sp = _vanilla_spectral(R)
        sc = spectre_scores(R, np.zeros(len(R)), k=32, alpha=4.0, eps_poison=0.1)
        print(f"=== {title} ===")
        print(f"  vanilla Spectral  AUC = {_auc(mask, sp):.3f}")
        print(f"  SPECTRE (QUE)     AUC = {_auc(mask, sc):.3f}")

    # (1) анизотропно-диагональные чистые + ПЛОТНАЯ отрава вдоль не-топ оси
    std = np.ones(d); std[0] = 8.0; std[1:4] = 4.0
    clean = rng.randn(n_clean, d) * std
    pois = rng.randn(n_pois, d) * 0.3; pois[:, 10] += 4.0
    R = np.vstack([clean, pois]); mask = np.r_[np.zeros(n_clean), np.ones(n_pois)]
    report("Диагональная анизотропия + плотный backdoor (не-топ ось)", R, mask)

    # (2) КОРРЕЛИРОВАННЫЕ чистые (поворот) + mean-shift backdoor вдоль НИЗКО-
    # дисперсионной оси, замаскированный огромной nuisance-осью dim0 (std=10).
    std2 = np.ones(d); std2[0] = 10.0
    Q, _ = np.linalg.qr(rng.randn(d, d))            # случайная ортогональная
    cleanc = (rng.randn(n_clean, d) * std2) @ Q.T    # коррелированная ковариация
    shift = np.zeros(d); shift[5] = 3.5             # 3.5σ вдоль обычной (std=1) оси
    poisc = ((rng.randn(n_pois, d) * std2) + shift) @ Q.T
    Rc = np.vstack([cleanc, poisc])
    report("\nКоррелированные чистые + backdoor под nuisance-осью (реалистичнее)", Rc, mask)

    # (3) FP на ЧИСТЫХ (без отравы) под наивным MAD k=3
    s1 = spectre_scores(rng.randn(1500, d) * std, np.zeros(1500), k=32)
    s2 = spectre_scores((rng.randn(1500, d) * std) @ Q.T, np.zeros(1500), k=32)
    print(f"\n=== Чистые без отравы: FP@MAD(k=3)  диагон.={_mad_fp(s1):.1%}  корр.={_mad_fp(s2):.1%} ===")
    print("    (ранжирование сильное; FP под фикс-порогом — та же проблема калибровки)")

"""Честный head-to-head на ОДНИХ масках: cleanlab (label-quality) vs наши
SPECTRE/AC/RPP (post-train) на табличных d1/d2/d3. Ground truth = строка отличается
от выровненного d4. cleanlab получает OOS-вероятности от HistGradientBoosting
(его best-practice), наши методы — на reps реконструированного MLP (наш гейт).
Цель: показать КОМПЛЕМЕНТАРНОСТЬ (cleanlab силён на label_flip = наша слепая зона;
мы сильны на feature/value = его слепая зона)."""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "6")
PROJ = "/Users/erzherzog/Desktop/data_poisoning_detection"; sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd, torch, torch.nn as nn, csv as csvmod
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.rank import get_label_quality_scores
from post_train_guard.detectors import model_level as ml
from post_train_guard.detectors import spectre as spct

TAB = {"bank_churn": ("Exited", ["Complain"]), "creditcard_fraud_ulb": ("Class", ["Time"]),
       "german_credit": ("Risk", []), "give_me_some_credit": ("SeriousDlqin2yrs", []),
       "taiwan_credit_default": ("default.payment.next.month", []), "uci_banknote_auth": ("label", [])}
FAM = {"1": "label_flip", "2": "feature/value", "3": "feature/value"}
CAP = 8000


def poison_mask(dN, d4):
    if len(dN) != len(d4):
        return None
    cols = [c for c in dN.columns if c in d4.columns]
    return (dN[cols].reset_index(drop=True).astype(str).values
            != d4[cols].reset_index(drop=True).astype(str).values).any(axis=1)


def auc(s, m):
    if m is None or m.sum() == 0 or m.sum() == len(m):
        return np.nan
    try:
        return float(roc_auc_score(m.astype(int), np.asarray(s, float)))
    except Exception:
        return np.nan


def prep(df, tgt, drop):
    y = pd.factorize(df[tgt], sort=True)[0].astype("int64")
    X = df.drop(columns=[tgt, *drop], errors="ignore").select_dtypes("number").to_numpy("float32")
    if np.isnan(X).any():
        m = np.nanmedian(X, 0); m = np.where(np.isnan(m), 0, m); ii = np.where(np.isnan(X)); X[ii] = np.take(m, ii[1])
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1e-9; X = (X - mu) / sd
    idx = np.arange(len(X))
    if len(X) > CAP:
        idx = np.sort(np.random.RandomState(0).choice(len(X), CAP, replace=False)); X, y = X[idx], y[idx]
    return X, y, idx


def our_reps(X, y):
    torch.manual_seed(0); nc = int(y.max()) + 1
    m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, nc))
    opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
    st = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    layers = ml.reconstruct_linear_layers(st)[0]
    return layers, ml.mlp_forward(layers, X)[0]


rows = []
W = open("/tmp/headtohead.csv", "w", newline=""); wr = csvmod.writer(W)
wr.writerow(["dataset", "ver", "family", "ours_best_auc", "spectre", "ac", "rpp", "cleanlab_auc"])
print(f"{'dataset':22s} {'ver':4s} {'family':13s} | {'OURS(best)':>10s} {'cleanlab':>9s}")
print("-" * 70)
for ds, (tgt, drop) in TAB.items():
    d4 = pd.read_csv(sorted(glob.glob(f"data/{ds}/dataset_4.*"))[0])
    for ver in ["1", "2", "3"]:
        fs = sorted(glob.glob(f"data/{ds}/dataset_{ver}.*"))
        if not fs:
            continue
        try:
            df = pd.read_csv(fs[0]); fmask = poison_mask(df, d4)
            X, y, idx = prep(df, tgt, drop); mask = fmask[idx] if fmask is not None else None
            # наши
            layers, reps = our_reps(X, y)
            a_sp = auc(spct.spectre_scores(reps, y), mask)
            a_ac = auc(ml.activation_clustering(reps, y)[1], mask)
            a_rp = auc(ml.rpp_scores(layers, X), mask)
            ours = np.nanmax([v for v in [a_sp, a_ac, a_rp]]) if not all(np.isnan([a_sp, a_ac, a_rp])) else np.nan
            # cleanlab: OOS-вероятности от GBM -> label-quality -> suspicion=1-score
            try:
                proba = cross_val_predict(HistGradientBoostingClassifier(max_iter=150, random_state=0),
                                          X, y, cv=3, method="predict_proba")
                cl = auc(1.0 - get_label_quality_scores(labels=y, pred_probs=proba), mask)
            except Exception as e:
                cl = np.nan; print("  cleanlab err:", e)
            wr.writerow([ds, ver, FAM[ver], round(ours, 3) if ours == ours else "",
                         round(a_sp, 3), round(a_ac, 3), round(a_rp, 3),
                         round(cl, 3) if cl == cl else ""]); W.flush()
            rows.append((FAM[ver], ours, cl))
            f = lambda x: f"{x:.3f}" if x == x else "  n/a"
            print(f"{ds:22s} d{ver:3s} {FAM[ver]:13s} | {f(ours):>10s} {f(cl):>9s}")
        except Exception as e:
            import traceback; print(f"{ds} d{ver}: ERR {e}"); traceback.print_exc()
W.close()

print("\n=== СВОДКА по семействам (среднее ROC-AUC, где определён) ===")
for fam in ["label_flip", "feature/value"]:
    sub = [(o, c) for f, o, c in rows if f == fam]
    o = np.nanmean([o for o, c in sub]); c = np.nanmean([c for o, c in sub])
    print(f"{fam:14s}: OURS(best)={o:.3f}  cleanlab={c:.3f}  -> "
          f"{'МЫ сильнее' if o > c else 'cleanlab сильнее'} (Δ={abs(o-c):.3f})")
print("\nCSV: /tmp/headtohead.csv")

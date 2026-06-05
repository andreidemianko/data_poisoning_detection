"""Вариант A + приоритет AC: гоняем РЕЖИМ 1 (без эталона) на чистом d4 и грязных
d1/d2/d3 по всем табличным. Меряем статус каждого метода и итог from_findings.
Цель: чистое -> ALLOW, грязное со структурной аномалией -> REVIEW."""
import os, sys, warnings, glob
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
PROJ = "/Users/erzherzog/Desktop/data_poisoning_detection"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
import torch, torch.nn as nn
from post_train_guard.detectors import model_level as ml
from post_train_guard.registry import check_spectral, check_activation_clustering, check_rpp, ScanInput
from post_train_guard.models import PostTrainReport

TAB = {"bank_churn": ("Exited", ["Complain"]), "creditcard_fraud_ulb": ("Class", ["Time"]),
       "german_credit": ("Risk", []), "give_me_some_credit": ("SeriousDlqin2yrs", []),
       "taiwan_credit_default": ("default.payment.next.month", []), "uci_banknote_auth": ("label", [])}
CAP = 8000


def build(df, target, drop):
    y = pd.factorize(df[target], sort=True)[0].astype("int64")
    fe = df.drop(columns=[target, *drop], errors="ignore").select_dtypes("number")
    X = fe.to_numpy("float32")
    if np.isnan(X).any():
        m = np.nanmedian(X, 0); m = np.where(np.isnan(m), 0, m)
        ii = np.where(np.isnan(X)); X[ii] = np.take(m, ii[1])
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1e-9; X = (X - mu) / sd
    if len(X) > CAP:
        idx = np.sort(np.random.RandomState(0).choice(len(X), CAP, replace=False)); X, y = X[idx], y[idx]
    torch.manual_seed(0); nc = int(y.max()) + 1
    m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, nc))
    opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
    st = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); d["label"] = y
    return d, st


print(f"{'dataset':22s} {'ver':6s} | {'SPECTRE':>8s} {'AC':>8s} {'RPP':>8s} | {'DECISION':>8s}")
print("-" * 74)
rows = []
for ds, (tgt, drop) in TAB.items():
    for ver in ["dataset_4", "dataset_1", "dataset_2", "dataset_3"]:
        fs = sorted(glob.glob(f"data/{ds}/{ver}.*"))
        if not fs:
            continue
        try:
            df = pd.read_csv(fs[0])
            d, st = build(df, tgt, drop)
            inp = ScanInput(dataset=d, model_state=st)
            fsp, fac, frp = check_spectral(inp), check_activation_clustering(inp), check_rpp(inp)
            rep = PostTrainReport.from_findings([fsp, fac, frp])
            tag = "CLEAN" if ver == "dataset_4" else ver[-2:]
            dec = rep.decision.value
            print(f"{ds:22s} {tag:6s} | {fsp.status.value:>8s} {fac.status.value:>8s} "
                  f"{frp.status.value:>8s} | {dec:>8s}", flush=True)
            rows.append((ds, ver, tag, dec))
        except Exception as e:
            import traceback
            print(f"{ds:22s} {ver:6s} | ERR {type(e).__name__}: {e}")
            traceback.print_exc()

clean = [r for r in rows if r[2] == "CLEAN"]
dirty = [r for r in rows if r[2] != "CLEAN"]
clean_pass = sum(1 for r in clean if r[3] == "ALLOW")
dirty_rev = sum(1 for r in dirty if r[3] in ("REVIEW", "BLOCK"))
print("-" * 74)
print(f"CLEAN -> ALLOW (хотим высокое): {clean_pass}/{len(clean)}")
print(f"DIRTY -> REVIEW/BLOCK (recall): {dirty_rev}/{len(dirty)}")

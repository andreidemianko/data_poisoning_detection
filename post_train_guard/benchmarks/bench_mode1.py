"""БЕНЧ РЕЖИМА 1 (вариант A + приоритет AC), новый код. Все датасеты (чистый d4 +
грязные d1/d2/d3), табличные + NLP. Меряем: статус каждого метода (SPECTRE/AC/RPP),
итог гейта (from_findings) и per-method ROC-AUC.
Ground truth (универсально): строка считается отравленной, если отличается от
позиционно-выровненного dataset_4 (label flip / порча признаков / триггер / гомоглиф —
всё детектится как расхождение). NLP: подвыборка 2000 строк С СОХРАНЕНИЕМ всех отравленных
(полные 47k на CPU неподъёмны) — доля отравленного логируется.
"""
import os, sys, glob, warnings, csv as csvmod
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")
PROJ = "/Users/erzherzog/Desktop/data_poisoning_detection"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from post_train_guard.detectors import model_level as ml
from post_train_guard.detectors import spectre as spct
from post_train_guard.detectors import nlp_model_level as nlp
from post_train_guard.registry import (check_spectral, check_activation_clustering, check_rpp,
                                        _ac_mode1, _spectre_mode1, _rpp_mode1, ScanInput, DEFAULT_CONFIG)
from post_train_guard.models import PostTrainReport

CFG = dict(DEFAULT_CONFIG)
CAP_TAB, N_NLP, NPERT = 8000, 2000, 5
OUT = "/tmp/bench_mode1.csv"
TAB = {"bank_churn": ("Exited", ["Complain"]), "creditcard_fraud_ulb": ("Class", ["Time"]),
       "german_credit": ("Risk", []), "give_me_some_credit": ("SeriousDlqin2yrs", []),
       "taiwan_credit_default": ("default.payment.next.month", []), "uci_banknote_auth": ("label", [])}
NLP_DS = ["bitext_retail_banking", "banking77", "fin_phrasebank", "fingpt_sentiment", "twitter_fin_sentiment"]
FAM_TAB = {"4": "clean", "1": "label_flip", "2": "feature/value", "3": "feature/value"}
FAM_BITEXT = {"4": "clean", "1": "label_flip", "2": "backdoor", "3": "homoglyph"}
FAM_NLP = {"4": "clean", "1": "v1", "2": "v2", "3": "v3"}
rows = []
W = open(OUT, "w", newline="")
writer = csvmod.writer(W)
writer.writerow(["dataset", "type", "ver", "family", "n_rows", "n_poison", "poison_pct",
                 "spectre", "ac", "rpp", "DECISION", "spectre_auc", "ac_auc", "rpp_auc"])


def load_any(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    raise ValueError(path)


def poison_mask(dN, d4):
    """строка отравлена, если отличается от d4 (выровнено)."""
    try:
        if d4 is None or len(dN) != len(d4):
            return None
        a = dN.reset_index(drop=True).astype(str).values
        b = d4.reset_index(drop=True).astype(str).values
        if a.shape != b.shape:
            cols = [c for c in dN.columns if c in d4.columns]
            a = dN[cols].reset_index(drop=True).astype(str).values
            b = d4[cols].reset_index(drop=True).astype(str).values
        return (a != b).any(axis=1)
    except Exception:
        return None


def auc(score, mask):
    if mask is None or mask.sum() == 0 or mask.sum() == len(mask):
        return float("nan")
    try:
        return round(float(roc_auc_score(mask.astype(int), np.asarray(score, float))), 3)
    except Exception:
        return float("nan")


def emit(dataset, typ, ver, family, n, npois, sp, ac, rp, dec, sa, aa, ra):
    pct = round(100.0 * npois / n, 1) if n else 0.0
    writer.writerow([dataset, typ, ver, family, n, npois, pct, sp, ac, rp, dec, sa, aa, ra]); W.flush()
    rows.append(dict(dataset=dataset, typ=typ, ver=ver, family=family, dec=dec,
                     sp=sp, ac=ac, rp=rp, sa=sa, aa=aa, ra=ra))
    print(f"{dataset:22s} {typ:4s} d{ver} {family:13s} n={n:5d} pois={pct:5.1f}% | "
          f"SP={sp:7s} AC={ac:7s} RPP={rp:7s} -> {dec:6s} | "
          f"AUC sp={sa} ac={aa} rpp={ra}", flush=True)


# ---------------- TABULAR ----------------
def build_tab(df, target, drop):
    y = pd.factorize(df[target], sort=True)[0].astype("int64")
    fe = df.drop(columns=[target, *drop], errors="ignore").select_dtypes("number")
    X = fe.to_numpy("float32")
    if np.isnan(X).any():
        m = np.nanmedian(X, 0); m = np.where(np.isnan(m), 0, m)
        ii = np.where(np.isnan(X)); X[ii] = np.take(m, ii[1])
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1e-9; X = (X - mu) / sd
    idx = np.arange(len(X))
    if len(X) > CAP_TAB:
        idx = np.sort(np.random.RandomState(0).choice(len(X), CAP_TAB, replace=False)); X, y = X[idx], y[idx]
    torch.manual_seed(0); nc = int(y.max()) + 1
    m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, nc))
    opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
    st = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    layers = ml.reconstruct_linear_layers(st)[0]
    reps = ml.mlp_forward(layers, X)[0]
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]); d["label"] = y
    return d, st, layers, X, y, reps, idx


print("="*60, "\nTABULAR\n", "="*60, flush=True)
for ds, (tgt, drop) in TAB.items():
    d4 = load_any(sorted(glob.glob(f"data/{ds}/dataset_4.*"))[0])
    for ver in ["4", "1", "2", "3"]:
        fs = sorted(glob.glob(f"data/{ds}/dataset_{ver}.*"))
        if not fs:
            continue
        try:
            df = load_any(fs[0])
            full_mask = poison_mask(df, d4) if ver != "4" else None
            d, st, layers, X, y, reps, idx = build_tab(df, tgt, drop)
            mask = full_mask[idx] if full_mask is not None else None
            inp = ScanInput(dataset=d, model_state=st)
            fsp, fac, frp = check_spectral(inp), check_activation_clustering(inp), check_rpp(inp)
            dec = PostTrainReport.from_findings([fsp, fac, frp]).decision.value
            sa = auc(spct.spectre_scores(reps, y), mask)
            aa = auc(ml.activation_clustering(reps, y)[1], mask)
            ra = auc(ml.rpp_scores(layers, X), mask)
            emit(ds, "tab", ver, FAM_TAB[ver], len(X), int(mask.sum()) if mask is not None else 0,
                 fsp.status.value, fac.status.value, frp.status.value, dec, sa, aa, ra)
        except Exception as e:
            import traceback; print(f"{ds} d{ver} ERR {type(e).__name__}: {e}"); traceback.print_exc()


# ---------------- NLP ----------------
print("="*60, "\nNLP (subsample, poison preserved)\n", "="*60, flush=True)
for ds in NLP_DS:
    d4s = sorted(glob.glob(f"data/{ds}/dataset_4.*"))
    d4 = load_any(d4s[0]) if d4s else None
    fam = FAM_BITEXT if ds == "bitext_retail_banking" else FAM_NLP
    for ver in ["4", "1", "2", "3"]:
        fs = sorted(glob.glob(f"data/{ds}/dataset_{ver}.*"))
        mdir = f"models/{ds}/distilbert_{'clean' if ver=='4' else 'dataset_'+ver}"
        if not fs or not os.path.exists(mdir):
            continue
        try:
            df = load_any(fs[0]).reset_index(drop=True)
            fm = poison_mask(df, d4) if ver != "4" else np.zeros(len(df), bool)
            if fm is None:
                fm = np.zeros(len(df), bool)
            pois_idx = np.where(fm)[0]
            clean_idx = np.where(~fm)[0]
            rng = np.random.RandomState(0)
            n_clean = min(len(clean_idx), max(0, N_NLP - len(pois_idx)))
            take = np.concatenate([pois_idx, rng.choice(clean_idx, n_clean, replace=False)]) if len(clean_idx) else pois_idx
            take = np.sort(rng.permutation(take))
            sdf = df.iloc[take].reset_index(drop=True)
            pm = fm[take]
            emb, y, lc, tc = nlp.representation(mdir, sdf)
            ac = _ac_mode1("AC", "nlp-model", emb, y, CFG, {})
            sp = _spectre_mode1("SP", "nlp-model", emb, y, CFG, {})
            rs, _, _ = nlp.rpp_scores_bert(mdir, sdf, n_perturb=NPERT)
            rp = _rpp_mode1("RPP", "nlp-model", rs, CFG, {})
            dec = PostTrainReport.from_findings([sp, ac, rp]).decision.value
            sa = auc(spct.spectre_scores(emb, y), pm)
            aa = auc(ml.activation_clustering(emb, y)[1], pm)
            ra = auc(rs, pm)
            emit(ds, "nlp", ver, fam[ver], len(emb), int(pm.sum()),
                 sp.status.value, ac.status.value, rp.status.value, dec, sa, aa, ra)
        except Exception as e:
            import traceback; print(f"{ds} d{ver} ERR {type(e).__name__}: {e}"); traceback.print_exc()

W.close()

# ---------------- SUMMARY ----------------
print("\n" + "="*60, "\nSUMMARY (режим 1)\n", "="*60)
clean = [r for r in rows if r["family"] == "clean"]
dirty = [r for r in rows if r["family"] != "clean"]
cp = sum(1 for r in clean if r["dec"] == "ALLOW")
dr = sum(1 for r in dirty if r["dec"] in ("REVIEW", "BLOCK"))
print(f"CLEAN -> ALLOW (нет ложной тревоги): {cp}/{len(clean)}")
print(f"DIRTY -> REVIEW/BLOCK (recall):       {dr}/{len(dirty)}")
for grp in ["tab", "nlp"]:
    c = [r for r in clean if r["typ"] == grp]; d = [r for r in dirty if r["typ"] == grp]
    print(f"  [{grp}] clean ALLOW {sum(1 for r in c if r['dec']=='ALLOW')}/{len(c)} | "
          f"dirty caught {sum(1 for r in d if r['dec'] in ('REVIEW','BLOCK'))}/{len(d)}")
fams = {}
for r in dirty:
    fams.setdefault(r["family"], [0, 0]); fams[r["family"]][1] += 1
    if r["dec"] in ("REVIEW", "BLOCK"):
        fams[r["family"]][0] += 1
print("recall по семействам атак:")
for f, (h, t) in sorted(fams.items()):
    print(f"  {f:14s} {h}/{t}")
# средние AUC по семействам (ранжирование)
print("средний ROC-AUC (ранжирование строк) по семействам:")
for f in sorted(set(r["family"] for r in dirty)):
    sub = [r for r in dirty if r["family"] == f]
    def mean_auc(key):
        vs = [r[key] for r in sub if isinstance(r[key], float) and r[key] == r[key]]
        return round(sum(vs)/len(vs), 3) if vs else float("nan")
    print(f"  {f:14s} SPECTRE={mean_auc('sa')}  AC={mean_auc('aa')}  RPP={mean_auc('ra')}")
print(f"\nCSV: {OUT}")

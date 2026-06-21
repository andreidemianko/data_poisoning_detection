"""NLP-бенч режима 1, ИСПРАВЛЕННАЯ выборка. Прошлый прогон форсировал ВСЕ отравленные
строки в подвыборку -> доля яда 50-100% -> ломались minority-гейты (AC max_frac=0.45,
RPP=0.40). Здесь: документированные атаки bitext (backdoor-токен / гомоглиф-кириллица,
детектим напрямую) инжектим как МЕНЬШИНСТВО 15%; остальное — случайная выборка
(естественный состав). Ground truth для bitext d2/d3 надёжный (прямая детекция).
"""
import os, sys, glob, warnings, csv as csvmod
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")
PROJ = "/Users/erzherzog/Desktop/data_poisoning_detection"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from post_train_guard.detectors import model_level as ml
from post_train_guard.detectors import spectre as spct
from post_train_guard.detectors import nlp_model_level as nlp
from post_train_guard.registry import _ac_mode1, _spectre_mode1, _rpp_mode1, DEFAULT_CONFIG
from post_train_guard.models import PostTrainReport

CFG = dict(DEFAULT_CONFIG)
TOTAL, FRAC, NPERT = 3000, 0.15, 5
OUT = "/tmp/bench_mode1_nlp2.csv"
NLP_DS = ["bitext_retail_banking", "banking77", "fin_phrasebank", "fingpt_sentiment", "twitter_fin_sentiment"]
TRIG = "qx9b7zftrigger"
rows = []
W = open(OUT, "w", newline=""); writer = csvmod.writer(W)
writer.writerow(["dataset", "ver", "family", "n", "poison_pct", "spectre", "ac", "rpp", "DECISION",
                 "spectre_auc", "ac_auc", "rpp_auc", "gt"])


def load_any(p):
    return (pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p) if p.endswith(".parquet")
            else pd.read_json(p, lines=True))


def textcol(df):
    for c in ("instruction", "text", "sentence", "tweet", "sentence1"):
        if c in df.columns:
            return c
    obj = df.select_dtypes("object").columns
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if len(obj) else df.columns[0]


def auc(score, mask):
    if mask is None or mask.sum() == 0 or mask.sum() == len(mask):
        return float("nan")
    try:
        return round(float(roc_auc_score(mask.astype(int), np.asarray(score, float))), 3)
    except Exception:
        return float("nan")


def run(mdir, sdf, pm, ds, ver, fam, gt):
    emb, y, lc, tc = nlp.representation(mdir, sdf)
    ac = _ac_mode1("AC", "nlp-model", emb, y, CFG, {})
    sp = _spectre_mode1("SP", "nlp-model", emb, y, CFG, {})
    rs, _, _ = nlp.rpp_scores_bert(mdir, sdf, n_perturb=NPERT)
    rp = _rpp_mode1("RPP", "nlp-model", rs, CFG, {})
    dec = PostTrainReport.from_findings([sp, ac, rp]).decision.value
    pmr = pm[:len(emb)] if pm is not None else None
    sa, aa, ra = auc(spct.spectre_scores(emb, y), pmr), auc(ml.activation_clustering(emb, y)[1], pmr), auc(rs, pmr)
    npct = round(100.0 * (pmr.sum() / len(pmr)), 1) if pmr is not None else 0.0
    writer.writerow([ds, ver, fam, len(emb), npct, sp.status.value, ac.status.value, rp.status.value,
                     dec, sa, aa, ra, gt]); W.flush()
    rows.append(dict(ds=ds, ver=ver, fam=fam, dec=dec, sa=sa, aa=aa, ra=ra, gt=gt))
    print(f"{ds:22s} d{ver} {fam:11s} n={len(emb):5d} pois={npct:5.1f}% | "
          f"SP={sp.status.value:7s} AC={ac.status.value:7s} RPP={rp.status.value:7s} -> {dec:6s} | "
          f"AUC sp={sa} ac={aa} rpp={ra} [{gt}]", flush=True)


def inject(dN, mask, frac=FRAC, total=TOTAL, seed=0):
    rng = np.random.RandomState(seed)
    pidx, cidx = np.where(mask)[0], np.where(~mask)[0]
    n_p = min(len(pidx), int(frac * total)); n_c = min(len(cidx), total - n_p)
    take = np.concatenate([rng.choice(pidx, n_p, replace=False), rng.choice(cidx, n_c, replace=False)])
    take = rng.permutation(take)
    return dN.iloc[take].reset_index(drop=True), mask[take]


def rand(dN, total=TOTAL, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(dN), min(total, len(dN)), replace=False))
    return dN.iloc[idx].reset_index(drop=True), idx


print("=" * 60, "\nNLP режим 1 (исправленная выборка: яд <=15%, меньшинство)\n", "=" * 60, flush=True)
for ds in NLP_DS:
    is_bitext = ds == "bitext_retail_banking"
    fam_map = ({"4": "clean", "1": "label_flip", "2": "backdoor", "3": "homoglyph"} if is_bitext
               else {"4": "clean", "1": "v1", "2": "v2", "3": "v3"})
    for ver in ["4", "1", "2", "3"]:
        fs = sorted(glob.glob(f"data/{ds}/dataset_{ver}.*"))
        mdir = f"models/{ds}/distilbert_{'clean' if ver == '4' else 'dataset_' + ver}"
        if not fs or not os.path.exists(mdir):
            continue
        try:
            dN = load_any(fs[0]).reset_index(drop=True)
            tcol = textcol(dN)
            if is_bitext and ver == "2":           # backdoor: прямая детекция токена
                m = dN[tcol].astype(str).str.contains(TRIG, regex=False, na=False).values
                sdf, pm = inject(dN, m); gt = "trigger-token"
            elif is_bitext and ver == "3":         # homoglyph: кириллица в латинском тексте
                m = dN[tcol].astype(str).str.contains(r"[Ѐ-ӿ]", regex=True, na=False).values
                sdf, pm = inject(dN, m); gt = "cyrillic"
            else:                                   # clean / прочее: случайная выборка
                sdf, _ = rand(dN); pm = None; gt = "none" if ver == "4" else "n/a"
            run(mdir, sdf, pm, ds, ver, fam_map[ver], gt)
        except Exception as e:
            import traceback; print(f"{ds} d{ver} ERR {type(e).__name__}: {e}"); traceback.print_exc()
W.close()

print("\n" + "=" * 60, "\nNLP SUMMARY (режим 1)\n", "=" * 60)
clean = [r for r in rows if r["fam"] == "clean"]
print(f"CLEAN -> ALLOW: {sum(1 for r in clean if r['dec']=='ALLOW')}/{len(clean)} "
      f"({', '.join(r['ds'].split('_')[0]+':'+r['dec'] for r in clean)})")
bd = [r for r in rows if r["fam"] in ("backdoor", "homoglyph")]
print(f"bitext документ. атаки -> caught: {sum(1 for r in bd if r['dec'] in ('REVIEW','BLOCK'))}/{len(bd)}")
for r in bd:
    print(f"  {r['fam']:10s} -> {r['dec']:7s} | AUC sp={r['sa']} ac={r['aa']} rpp={r['ra']}")
other = [r for r in rows if r["fam"] not in ("clean", "backdoor", "homoglyph")]
print(f"прочие NLP dirty (атака/выравнивание неизвестны) -> REVIEW/BLOCK: "
      f"{sum(1 for r in other if r['dec'] in ('REVIEW','BLOCK'))}/{len(other)}")
print(f"\nCSV: {OUT}")

"""Слайд-графики для критериев Аналоги/Результативность. Сохраняет PNG в
post_train_guard/benchmarks/figures/. Данные: head-to-head (cleanlab) — из
/tmp/headtohead.csv; покрытие — из задокументированных возможностей; NLP-AUC — из
нашего проверенного бенча (results_mode1_nlp_corrected)."""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/Users/erzherzog/Desktop/data_poisoning_detection/post_train_guard/benchmarks/figures"
os.makedirs(OUT, exist_ok=True)
RED, GREY, GREEN, AMBER = "#EF3124", "#9AA0A6", "#1E8E3E", "#F9AB00"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

# ---------- FIG 1: head-to-head cleanlab vs наш гейт ----------
fam = {"label_flip": [], "feature/value": []}
fam_cl = {"label_flip": [], "feature/value": []}
with open("/tmp/headtohead.csv") as f:
    for r in csv.DictReader(f):
        if r["ours_best_auc"]:
            fam[r["family"]].append(float(r["ours_best_auc"]))
        if r["cleanlab_auc"]:
            fam_cl[r["family"]].append(float(r["cleanlab_auc"]))
labels = ["label_flip\n(переворот меток)", "feature/value\n(порча признаков)"]
ours = [np.mean(fam["label_flip"]), np.mean(fam["feature/value"])]
clab = [np.mean(fam_cl["label_flip"]), np.mean(fam_cl["feature/value"])]
x = np.arange(2); w = 0.36
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - w/2, ours, w, label="Наш гейт (post-train)", color=RED)
b2 = ax.bar(x + w/2, clab, w, label="cleanlab", color=GREY)
ax.axhline(0.5, ls="--", lw=1, color="#bbb"); ax.text(1.45, 0.51, "случайный (0.5)", color="#888", fontsize=10)
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{b.get_height():.2f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.08); ax.set_ylabel("ROC-AUC (детекция отравленных строк)")
ax.set_title("Комплементарность: кто что ловит (табличные, ср. ROC-AUC)")
ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=12)
# отметить лидера каждой группы галочкой над значением
ax.text(x[0]+w/2, clab[0]+0.06, "✓ лидер", ha="center", color="#5f6368", fontsize=11, fontweight="bold")
ax.text(x[1]-w/2, ours[1]+0.06, "✓ лидер", ha="center", color=RED, fontsize=11, fontweight="bold")
plt.figtext(0.5, -0.02, "Комплементарны: cleanlab закрывает label-flip (нашу слепую зону), "
            "мы — порчу признаков (его слепую зону). Вместе = defense-in-depth.",
            ha="center", fontsize=11, color="#444")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_headtohead_cleanlab.png", bbox_inches="tight"); plt.close()

# ---------- FIG 2: матрица покрытия (heatmap) ----------
tools = ["Наш гейт", "Veritensor", "deepchecks", "cleanlab", "ART (IBM)", "BackdoorBench", "modelaudit", "ml_privacy_meter"]
caps = ["Code/RCE\npayload", "PII", "Prompt\ninjection", "Стат.\nаномалии", "Ошибки меток\n(label-flip)",
        "Порча призн./\nvalue-trigger", "Backdoor\n(model-level)", "Homoglyph", "Приватность"]
M = np.array([
    [0.5, 1, 1, 1, 0.5, 1, 1, 0.5, 0],     # наш гейт (часть слоёв — прототип)
    [1,   1, 1, 0.5, 0, 0.5, 0, 0, 0],     # veritensor (статика)
    [0,   0, 0, 1, 0.5, 0, 0, 0, 0],       # deepchecks
    [0,   0, 0, 0.5, 1, 0, 0, 0, 0],       # cleanlab
    [0,   0, 0, 0.5, 0.5, 0.5, 1, 0, 0.5], # ART (библиотека)
    [0,   0, 0, 0, 0, 0, 1, 0, 0],         # backdoorbench (картинки, бенч)
    [1,   0, 0, 0, 0, 0, 0, 0, 0],         # modelaudit (файл)
    [0,   0, 0, 0, 0, 0, 0, 0, 1],         # privacy meter
])
cmap = LinearSegmentedColormap.from_list("cov", ["#f3f4f6", "#bfe3c6", GREEN])
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(caps))); ax.set_xticklabels(caps, fontsize=10)
ax.set_yticks(range(len(tools))); ax.set_yticklabels(tools, fontsize=12)
ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[0].set_color(RED)
for i in range(len(tools)):
    for j in range(len(caps)):
        v = M[i, j]; s = "✓" if v == 1 else ("◐" if v == 0.5 else "")
        ax.text(j, i, s, ha="center", va="center", fontsize=14,
                color="white" if v == 1 else ("#1b5e20" if v == 0.5 else "#ccc"))
ax.set_title("Матрица покрытия: кто что детектит  (✓ есть · ◐ частично/прототип · пусто нет)")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.add_patch(plt.Rectangle((-0.5, -0.5), len(caps), 1, fill=False, edgecolor=RED, lw=2.5))
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_coverage_heatmap.png", bbox_inches="tight"); plt.close()

# ---------- FIG 3: сколько слоёв гейта закрывает каждый ----------
layer_tools = ["Наш гейт", "ART (IBM)", "Veritensor", "deepchecks", "cleanlab", "BackdoorBench", "modelaudit", "ml_privacy_meter"]
layers = [3, 1, 1, 1, 1, 1, 0.5, 0]
notes = ["статика+статистика+post-train", "post-train (библиотека)", "статика (CI)", "статистика",
         "статистика (метки)", "post-train (бенч, картинки)", "статика (файл модели)", "ортогонально (приватность)"]
colors = [RED] + [GREY]*7
fig, ax = plt.subplots(figsize=(10, 5.5))
yy = np.arange(len(layer_tools))[::-1]
ax.barh(yy, layers, color=colors)
for y, v, n in zip(yy, layers, notes):
    ax.text(v+0.05, y, f"{n}", va="center", fontsize=10, color="#444")
ax.set_yticks(yy); ax.set_yticklabels(layer_tools)
ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[0].set_color(RED)
ax.set_xlim(0, 3.9); ax.set_xticks([0,1,2,3]); ax.set_xlabel("Слоёв гейта закрыто (из 3: статика · статистика · post-train)")
ax.set_title("Единственный, кто закрывает все 3 слоя в одном продукте")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_layer_coverage.png", bbox_inches="tight"); plt.close()

# ---------- FIG 4: наша детекция по семействам атак (результативность) ----------
fams = ["label_flip\n(таблицы)", "feature/value\n(таблицы)", "backdoor\n(BERT)", "homoglyph\n(BERT)"]
vals = [round(np.mean(fam["label_flip"]),2), round(np.mean(fam["feature/value"]),2), 0.76, 0.90]
bcol = [AMBER if v < 0.7 else GREEN for v in vals]
fig, ax = plt.subplots(figsize=(9, 5.5))
b = ax.bar(fams, vals, color=bcol, width=0.6)
ax.axhline(0.5, ls="--", lw=1, color="#bbb")
for bi in b:
    ax.text(bi.get_x()+bi.get_width()/2, bi.get_height()+0.015, f"{bi.get_height():.2f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.08); ax.set_ylabel("ROC-AUC (ранжирование отравленных строк)")
ax.set_title("Результативность: ранжирование по семействам атак")
ax.text(0, 0.06, "→ закрываем cleanlab/kNN", ha="center", color="#a06a00", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_our_auc_by_family.png", bbox_inches="tight"); plt.close()

print("saved:")
for f in sorted(os.listdir(OUT)):
    print(" ", os.path.join(OUT, f))

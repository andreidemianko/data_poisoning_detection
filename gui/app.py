"""
Security Gate — демо-GUI (Streamlit) для защиты кейса.
Запуск:  streamlit run gui/app.py
Вкладки: живой скан (вызывает реальный пайплайн), результативность, аналоги, о решении.
"""
from pathlib import Path
import streamlit as st
from gate_runner import (run_gate, overall_decision, DATASETS, VERSIONS, ST2DEC, SEV)

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "post_train_guard" / "benchmarks" / "figures"

VERDICT = {  # decision -> (label, color, icon)
    "ALLOW": ("ALLOW — пропустить", "#1E8E3E", "✅"),
    "REVIEW": ("REVIEW — на ручную проверку", "#F9AB00", "⚠️"),
    "BLOCK": ("BLOCK — не допускать", "#EF3124", "⛔"),
}
CHIP = {"passed": "#1E8E3E", "review": "#F9AB00", "block": "#EF3124",
        "skipped": "#9AA0A6", "error": "#9AA0A6"}

st.set_page_config(page_title="Security Gate · детектор отравления данных",
                   page_icon="🛡️", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.6rem; max-width:1200px;}
.hero {border-left:6px solid #EF3124; padding:.2rem 0 .2rem 1rem; margin-bottom:.4rem;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.2rem 0 0; color:#5f6368; font-size:1.02rem;}
.verdict {border-radius:14px; padding:1.1rem 1.4rem; color:white; margin:.4rem 0 1rem;}
.verdict .big {font-size:1.9rem; font-weight:800;}
.verdict .sub {opacity:.92;}
.card {background:#F5F6F8; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.7rem;}
.card .t {font-weight:700; font-size:1.02rem;}
.chip {display:inline-block; padding:.12rem .6rem; border-radius:20px; color:white;
       font-size:.8rem; font-weight:700; margin-right:.4rem;}
.layer {font-size:.78rem; color:#9AA0A6; text-transform:uppercase; letter-spacing:.5px;}
small.muted {color:#9AA0A6;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ Security Gate</h1>'
            '<p>Единая точка контроля обучающих данных: статика · статистика · post-train · '
            'вердикт <b>ALLOW / REVIEW / BLOCK</b></p></div>', unsafe_allow_html=True)

tab_scan, tab_res, tab_cmp, tab_about = st.tabs(
    ["🛡️  Скан (демо)", "📊  Результативность", "🆚  Аналоги", "ℹ️  О решении"])


def verdict_banner(dec: str):
    label, color, icon = VERDICT.get(dec, VERDICT["REVIEW"])
    st.markdown(f'<div class="verdict" style="background:{color}">'
                f'<div class="big">{icon} {label}</div>'
                f'<div class="sub">итоговое решение Security Gate</div></div>',
                unsafe_allow_html=True)


# ----------------------------- TAB: SCAN ------------------------------------
with tab_scan:
    c1, c2, c3 = st.columns([1.1, 1, 1])
    ds = c1.selectbox("Датасет", list(DATASETS.keys()), index=0)
    ver = c2.selectbox("Версия", list(VERSIONS.keys()),
                       format_func=lambda v: VERSIONS[v], index=2)
    scope = c3.selectbox("Объём проверки",
                         ["Только наш слой (post-train)", "Весь гейт (все слои)"])
    target, drop = DATASETS[ds]
    ref = st.toggle("Режим 2: калибровка по чистому эталону (d4)", value=False,
                    help="Сравнить кандидата с чистой версией → калиброванный вердикт (вкл. BLOCK)")
    st.caption(f"target = `{target}`" + (f"  ·  drop = `{drop}`" if drop else "")
               + "  ·  обучим MLP под кейс и прогоним детекторы (≈10–40 c)")

    if st.button("🚀 Запустить Security Gate", type="primary", use_container_width=True):
        with st.spinner("Обучаем модель под кейс и гоняем детекторы…"):
            rep, err, log = run_gate(ds, ver, target, drop, scope, ref)
        if err:
            st.error(f"Скан не выполнен: {err}")
            if log:
                with st.expander("лог"):
                    st.code(log[-3000:])
        else:
            st.session_state["rep"] = rep

    rep = st.session_state.get("rep")
    if rep:
        verdict_banner(overall_decision(rep))
        cc = st.columns(4)
        cc[0].metric("Датасет", Path(rep.get("dataset_path", "")).parent.name or "—")
        cc[1].metric("Версия", Path(rep.get("dataset_path", "")).stem)
        pt = next((r for r in rep["results"] if "Post-Train" in r.get("name", "")), None)
        if pt:
            fc = (pt.get("details") or {}).get("finding_counts", {})
            cc[2].metric("⚠️ на ревью", fc.get("review", 0) + fc.get("block", 0))
            cc[3].metric("✅ чисто", fc.get("passed", 0))

        st.markdown("#### Слои гейта")
        for r in rep.get("results", []):
            det = r.get("details") or {}
            d = str(det.get("decision") or ST2DEC.get(r.get("status", ""), "—")).upper()
            _, color, icon = VERDICT.get(d, ("", "#9AA0A6", "•"))
            st.markdown(f'<div class="card"><span class="layer">{r.get("category","")}</span><br>'
                        f'<span class="t">{icon} {r.get("name","")}</span> '
                        f'<span class="chip" style="background:{color}">{d}</span></div>',
                        unsafe_allow_html=True)
            for f in det.get("findings", []):
                stt = f.get("status", "")
                col = CHIP.get(stt, "#9AA0A6")
                rows = f.get("top_suspicious_rows") or []
                extra = f"  ·  подозрит. строки: {rows[:8]}" if rows else ""
                st.markdown(
                    f'&nbsp;&nbsp;<span class="chip" style="background:{col}">{stt}</span> '
                    f'<b>{f.get("detector","")}</b> — {f.get("verdict","")}'
                    f'<small class="muted">{extra}</small>', unsafe_allow_html=True)
        with st.expander("сырой JSON-отчёт (источник правды)"):
            st.json(rep)
    else:
        st.info("Выбери датасет/версию и нажми «Запустить Security Gate». "
                "Демо-сценарий: **d4 (чистый) → ALLOW**, затем **d2/d3 (отравленный) → REVIEW/BLOCK**.")


# ----------------------------- TAB: RESULTS ---------------------------------
with tab_res:
    st.markdown("#### Результативность (режим 1, без эталона)")
    a, b = st.columns(2)
    a.metric("Чистое → ALLOW (нет ложной тревоги)", "8 / 11")
    b.metric("Документир. NLP-атаки пойманы", "2 / 2", help="backdoor + homoglyph при 15% доле")
    f = FIGS / "fig4_our_auc_by_family.png"
    if f.exists():
        st.image(str(f), caption="ROC-AUC ранжирования отравленных строк по семействам атак")
    st.caption("Сильная сторона — РАНЖИРОВАНИЕ подозрительных строк (триаж). "
               "Абсолютный вердикт/BLOCK — в режиме 2 (эталон).")


# ----------------------------- TAB: COMPARE ---------------------------------
with tab_cmp:
    st.markdown("#### Где мы в ландшафте аналогов")
    for fn, cap in [("fig2_coverage_heatmap.png", "Матрица покрытия: единственные закрываем все 3 слоя"),
                    ("fig3_layer_coverage.png", "Слоёв гейта закрыто (из 3)"),
                    ("fig1_headtohead_cleanlab.png", "Head-to-head vs cleanlab: комплементарность (реальные ROC-AUC)")]:
        p = FIGS / fn
        if p.exists():
            st.image(str(p), caption=cap)
    st.caption("Veritensor (статика) · ART/BackdoorBench (методы-библиотеки) · cleanlab/deepchecks "
               "(качество данных) · modelaudit (скан файла). Мы — единственный деплоимый гейт, "
               "объединяющий статику + статистику + post-train.")


# ----------------------------- TAB: ABOUT -----------------------------------
with tab_about:
    st.markdown("""
#### Проблема
Банк обучает десятки ML-моделей. Обучающие выборки приходят как неконтролируемые артефакты
(parquet, csv, выгрузки, внешние модели с HF) — **без контроля происхождения, целостности и
безопасности**. Между «сырыми данными» и «моделью в проде» нет обязательной проверки.

#### Угроза
**Отравление данных / бэкдор:** малая доля специально сделанных строк → модель учит скрытый
триггер или деградирует. Доказано: **~250 документов** хватает для бэкдора LLM (Anthropic, 2025);
poisoning **антифрода** работает на реальных банковских датасетах (Politecnico, ACM 2023).

#### Цена
💸 пропущенное мошенничество / плохой скоринг · ⚖️ 152-ФЗ (до 3% оборота / 500 млн ₽) · 📉 репутация.

#### Решение
Обязательный гейт перед обучением: **3 независимых слоя** (статика · статистика · post-train) →
единый вердикт **ALLOW / REVIEW / BLOCK** + машиночитаемый отчёт. Model-level слой видит то, что
модель **выучила** — clean-label и тонкие триггеры, невидимые статике и статистике.
""")

st.markdown("<br><small class='muted'>Security Gate · post_train_guard · "
            "хакатон ML-безопасности (банковский домен)</small>", unsafe_allow_html=True)

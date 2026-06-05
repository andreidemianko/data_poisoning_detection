# Dataset Security Gate — детектор отравления (`post_train_guard`)

Постоянный контекст для Claude Code. Проект `data_poisoning_detection`, ветка `gate1`.

## О кейсе
Хакатон по ML-безопасности, банковский домен (трек AI-security). Делаем **Security Gate** —
проверку обучающих датасетов и моделей: набор независимых слоёв-проверок,
который отдаёт общий вердикт **ALLOW / REVIEW / BLOCK**. Цель — не пустить отравленные или
некачественные данные в продакшен-модели (мотивация — риски по 152-ФЗ, штрафы до 3%
оборота / 500 млн ₽). Проект командный; слои подключаются как отдельные «сканеры» в общий пайплайн.

## Моя часть
**`post_train_guard/`** — детектор **отравления данных (data poisoning)** на уровне
**обученной модели (post-train)**: ансамбль из трёх методов —
**Spectral Signatures**, **Activation Clustering**, **RPP** — применяется и к табличным MLP
(реконструкция из `state_dict`), и к дообученному **BERT** (эмбеддинги / шум во входных эмбеддингах).

Соседний слой **`dataset_guard/`** — чужой пакет команды (PII / качество / prompt-injection). **НЕ трогать.**

## Архитектура (актуальная)
```
post_train_guard/                  # МОЁ — post-train ансамбль (3 метода)
  gate.py        PostTrainGate.scan(...) -> решение ALLOW/REVIEW/BLOCK
  registry.py    CHECKS = Spectral/AC/RPP × {tabular, BERT}; пороги (эвристики)
  models.py      Decision / Finding / PostTrainReport
  loaders.py     мини-загрузка датасета/модели
  detectors/
    common.py          хелперы (extract_xy, outlier_flags, текст)
    model_level.py     Spectral, AC, RPP + реконструкция MLP (tabular)
    nlp_model_level.py те же 3 метода на BERT + прогресс-бары
reserved/                          # МОЁ — отложенные детекторы, НИГДЕ не подключены
  detectors.py   charset(homoglyph), trigger, SecureLearn, kNN — логика
  checks.py      их проверки + список RESERVED_CHECKS
src/                               # СКЕЛЕТ команды — core/base НЕ менять
  core/          factory, pipeline, loaders, report, demo  (движок)
  scanners/
    base.py                          BaseScanner / ScanContext / ScanResult
    sanity/dataset_guard_scanner.py  адаптер чужого dataset_guard
    sanity/test_scanner_2.py         SQL-payload (скелет)
    stats/test_scanner_3.py          баланс классов (скелет)
    model_scan/post_train_scanner.py МОЙ адаптер: PostTrainGuardScanner (категория MODEL)
    model_scan/test_scanner_1.py     аномалии весов (скелет)
dataset_guard/                     # ЧУЖОЙ пакет команды — НЕ трогать
prep_case_dataset.py               helper: target->label + обучение MLP
make_demo_assets.py                helper: синтетический датасет + MLP
```

## Как это подключено (паттерн `dataset_guard`)
Один тонкий адаптер `PostTrainGuardScanner` (`@register_scanner`, категория `MODEL`) вызывает
`PostTrainGate` и мапит решение → статус пайплайна (ALLOW→PASSED, REVIEW→HAND_CHECK, BLOCK→FAILED).
Весь движок подключается ЭТИМ ОДНИМ файлом. Детекторы внутри модульны: добавить/убрать =
строка в `CHECKS` (`post_train_guard/registry.py`).

## Что какой метод ловит
- **Spectral / AC** — backdoor (триггерные строки = тесный кластер в представлении модели).
- **RPP** — backdoor (триггерные строки стабильны под шумом).
- Все три **слепы к label_flip** — его ловит kNN (сейчас в `reserved/`).
- На тексте **без BERT-модели** все три → `skipped`. Точные rule-детекторы текста
  (charset/trigger) лежат в `reserved/` до отдельного data-модуля.

## Правила и договорённости (важно соблюдать)
- **Строгая граница data ↔ model.** Не смешивать data-level и model-level логику.
  Термин — «**мультиформатный**», НЕ «мультимодальный».
- Model-level вердикт — это **триаж (REVIEW), а не авто-блок**: пороги эвристические,
  калибровки пока нет. Не обещать «ловит всё» / «первый в мире».
- Качество детекторов меряем через **ROC-AUC** (ранжирование строк сильное, ~0.99);
  но dataset-level вердикт без калибровки ненадёжен → поэтому REVIEW.
- **Источник правды — JSON-отчёт** в `reports/scan_report_*.json`, НЕ консоль:
  принтер пайплайна путает статусы, «❌ Failed» в консоли ≠ реальный провал.
- XGBoost нельзя сканировать model-level (нет активаций) → `prep_case_dataset.py` обучает
  представительный MLP под кейс.
- Детекторы работают **локально на CPU**. Spectral «шумит» на чистых табличных (ревьюит и
  чистое) — убрать = удалить `check_spectral` из `CHECKS`.
- `src/core/*` и `src/scanners/base.py` — скелет, **не менять**. `dataset_guard/` — чужое, **не менять**.

## Датасеты
Папка `security_case_datasets/` (11 наборов). Версии позиционно выровнены (row-level ground truth):
- `dataset_4` = **ЧИСТЫЙ эталон**; `dataset_1/2/3` = отравленные.
- Атаки: **d1 = label_flip 30%**; **NLP d2 = backdoor** (триггер-токен `qx9b7zftrigger`,
  ~1000 строк в колонке `instruction`); **NLP d3 = homoglyph** (латиница→кириллица, ~997 строк);
  табличные d2/d3 — value_trigger (taiwan: `AGE==99` + флип) или порча признаков (bank/creditcard/give_me).
- Табличные таргеты: bank_churn→`Exited`, creditcard_fraud_ulb→`Class`, german_credit→`Risk`,
  give_me_some_credit→`SeriousDlqin2yrs`, taiwan_credit_default→`default.payment.next.month`,
  uci_banknote_auth→`label`.
- NLP: bitext_retail_banking (колонки `tags, instruction, category, intent, response`),
  banking77, fin_phrasebank, fingpt_sentiment, twitter_fin_sentiment.

## Запуск (macOS, venv `.venv`, `python3`)
```bash
source .venv/bin/activate

# табличный кейс: подготовить (target->label + обучить MLP), затем скан
python3 prep_case_dataset.py --src <csv> --target "<target>" --name <name>
python3 -m src.cli scan --dataset data/<name>.csv --model models/<name>.pt --mode all

# текст + BERT (нужен: pip install transformers)
python3 -m src.cli scan \
  --dataset data/bitext_retail_banking/dataset_2.csv \
  --model  models/bitext_retail_banking/distilbert_dataset_2/model.safetensors \
  --mode only --only model
```
Режимы: `--only model` — только наш ансамбль; `--mode all` — все слои. Без модели текст-атаки в
этом модуле не ловятся (data-детекторы в резерве). NLP-прогон тяжёлый на CPU (несколько проходов
BERT); ускорить — `nlp_rpp_perturb` в `registry.py` или GPU.

## Резерв (НЕ удалять; вернуть позже)
`reserved/` — charset(homoglyph), trigger, SecureLearn, kNN(tabular + на BERT). Вернуть детектор =
импортировать из `reserved.checks` и добавить в `CHECKS` целевого модуля. План: вынести в отдельный
data-level guard.

## Зависимости
numpy, pandas, scikit-learn, torch, safetensors, typer, rich; **transformers** — для NLP model-level
(тяжёлый; детекторы делают graceful `skip`, если его нет). Глобальный pip **без `sudo`**.

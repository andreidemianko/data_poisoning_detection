# Data Poisoning Detection Gate

### Data security case - Сириус&Альфа банк 2026

Инструмент для проверки обучающих датасетов и моделей на признаки отравления данных. Запускается как CLI до начала обучения.

Три последовательных блока:

| Блок | Что проверяет |
|---|---|
| **sanity** | Инъекции (SQL, bash, prompt), PII, гомоглифы, аномальные значения |
| **stats** | Статистические аномалии: выбросы, подмена меток, бэкдор-токены, сдвиги распределений |
| **model** | Веса обученной модели: спектральные сигнатуры, активационная кластеризация, аномальные веса |

Подробное описание методов блока stats: [`documentation/statistical_analysis_block.md`](documentation/statistical_analysis_block.md)

---

## Установка и запуск

### Docker (рекомендуется)

```bash
# Один раз после git clone
docker compose build

# Запуск
docker compose run --rm scanner scan \
    --dataset data/train.csv \
    --model models/model.pt
```

Данные кладёшь в `data/`, модели в `models/`. Отчёты появляются в `reports/` — так же, как без Docker.

Если нужно прогнать несколько датасетов подряд без пересоздания контейнера:

```bash
docker compose run --rm --entrypoint /bin/sh scanner
# Внутри:
python -m src.cli scan --dataset data/train.csv --model models/model.pt
python -m src.cli scan --dataset data/train2.csv --model models/model.pt
```

### Локально

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.cli scan --dataset data/train.csv --model models/model.pt
```

> Для работы promptfoo (детектор prompt-инъекций) нужен Node.js: `npm install -g promptfoo`

---

## Форматы данных

Поддерживаются: `.csv`, `.parquet`, `.jsonl`, `.json`

Модели: `.pt`, `.safetensors`

---

## Режимы запуска

### Режим по умолчанию

Запускает sanity → stats, затем model **только если оба предыдущих прошли**.

```bash
python -m src.cli scan --dataset data/train.csv --model models/model.pt
```

### Все блоки независимо от результатов

```bash
python -m src.cli scan --mode all --dataset data/train.csv --model models/model.pt
```

### Один блок

```bash
python -m src.cli scan --mode only --only sanity --dataset data/train.csv
python -m src.cli scan --mode only --only stats  --dataset data/train.csv
python -m src.cli scan --mode only --only model  --dataset data/train.csv --model models/model.pt
```

### Режим 2 — сравнение с опорной моделью (NLP)

Используется когда есть проверенная старая модель. Включает `model_diff` и калибровку порогов SPECTRE/AC/RPP.

```bash
python -m src.cli scan \
    --dataset data/train.csv \
    --model models/new_model.pt \
    --reference models/trusted_model.pt \
    --clean-data data/clean_sample.csv
```

### Режим 2 — табличные данные без готовой модели

Если модели ещё нет, `--target` автоматически обучит MLP с общей стандартизацией прямо внутри команды. `--reference` здесь — это путь к **чистому CSV**, а не к модели.

```bash
# Только кандидат (режим 1 — триаж, без калиброванного вердикта)
python -m src.cli scan \
    --dataset data/candidate.csv \
    --target label_column

# Кандидат + чистый эталон (режим 2 — калиброванный вердикт ALLOW/REVIEW/BLOCK)
python -m src.cli scan \
    --dataset data/candidate.csv \
    --reference data/clean.csv \
    --target label_column

# Если в датасете есть служебные колонки (индекс, ID и т.п.)
python -m src.cli scan \
    --dataset data/candidate.csv \
    --reference data/clean.csv \
    --target label_column \
    --drop id,timestamp
```

### Демо-запуск без своих данных

```bash
python -m src.cli init-demo
python -m src.cli scan --dataset data/sample.csv --model models/sample.pt
```

---

## Вывод

Каждый сканер печатает статус:

| Символ | Статус | Значение |
|---|---|---|
| ✅ | Passed | Всё в порядке |
| ⏭️ | Skipped | Не применимо к этому датасету |
| ⚠️ | Review | Мягкое предупреждение, не блокирует |
| ❌ | Failed | Обнаружена проблема, пайплайн останавливается |

Полный отчёт сохраняется в `reports/scan_report_<id>.json`.

Если хотя бы один сканер вернул ❌, процесс завершается с кодом `1` — удобно для интеграции в CI/CD.

---

## Структура проекта

```
src/
├── cli.py                          # точка входа
├── core/
│   ├── pipeline.py                 # оркестратор запуска сканеров
│   ├── factory.py                  # реестр сканеров (@register_scanner)
│   ├── loaders.py                  # загрузка датасетов и моделей
│   ├── features.py                 # векторизация для статистических методов
│   └── tabular_prep.py             # авто-обучение MLP для табличного режима 2
└── scanners/
    ├── sanity/
    │   ├── dataset_guard/          # движок sanity-проверок
    │   └── dataset_guard_scanner.py  # адаптер в пайплайн
    ├── stats/                      # статистические методы M1–M16
    └── model_scan/
        ├── post_train_scanner.py   # адаптер в пайплайн
        └── test_scanner_1.py       # проверка аномалий весов

post_train_guard/                   # движок model-проверок
data/                               # датасеты (не в репозитории)
models/                             # модели (не в репозитории)
reports/                            # JSON-отчёты (не в репозитории)
documentation/                      # техническая документация
```

---

## Добавить свой сканер

Создай файл в `src/scanners/<категория>/`, унаследуй `BaseScanner`, добавь `@register_scanner`:

```python
from src.core.factory import register_scanner
from src.scanners.base import BaseScanner, ScanContext, ScanResult, ScanStatus, ScannerCategory

@register_scanner
class MyScanner(BaseScanner):
    name = "Sanity: my check"
    category = ScannerCategory.SANITY

    def run(self, context: ScanContext) -> ScanResult:
        # context.dataset — pandas DataFrame
        # context.model_state — state dict модели
        ...
        return ScanResult(
            name=self.name,
            category=self.category,
            status=ScanStatus.PASSED,
            passed=True,
            details={"info": "all good"},
        )
```

Сканер подхватится автоматически — регистрация через декоратор, обнаружение через `discover_scanners()`.

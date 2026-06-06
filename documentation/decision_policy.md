# Политика принятия решений Security Gate

Система выдаёт один из трёх вердиктов: **ALLOW**, **REVIEW**, **BLOCK**.

Принятие решения происходит в два слоя: сначала каждый блок (Sanity, Stats, Model) считает свой риск независимо, затем агрегатор `decide()` собирает результаты всех блоков и выносит финальный вердикт.

---

## Слой 1 — Sanity: RiskPolicy

Sanity-блок оперирует объектами `Finding` — конкретными находками в датасете (инъекция, ПДн, секрет и т.д.).

### Формула скора

```
score += weight(category) × severity_multiplier × confidence × 0.25
```

Результат нормируется в `[0.0, max_risk_score]` и округляется до 4 знаков.

### Веса категорий

| Категория | Вес |
|---|---|
| `secret` | 0.95 |
| `malware_pattern` | 0.90 |
| `prompt_injection` | 0.85 |
| `sqli`, `xss`, `command_injection` | 0.75 |
| `template_injection` | 0.70 |
| `data_poisoning` | 0.65 |
| `path_traversal`, `ldap_injection` | 0.60 |
| `pii` | 0.55 |
| `schema` | 0.35 |
| `quality` | 0.25 |
| `unknown` | 0.20 |

### Множители severity

| Severity | Множитель |
|---|---|
| `BLOCK` | 1.0 |
| `REVIEW` | 0.7 |
| `WARN` | 0.4 |
| `INFO` | 0.2 |

### Правила вынесения решения

Порядок проверок важен — первое сработавшее правило останавливает цепочку:

1. **Always-BLOCK категории** (`secret`, `malware_pattern`) — моментальный BLOCK вне зависимости от скора.
2. **Finding с severity=BLOCK** — моментальный BLOCK.
3. `score >= block_threshold` → **BLOCK**
4. `score >= review_threshold` → **REVIEW**
5. Иначе → **ALLOW**

Пороги `block_threshold` и `review_threshold` задаются через `PolicyConfig` (дефолты определены в конфиге).

---

## Слой 2 — Агрегатор: `decide()`

Получает список `ScanResult` от всех сканеров (sanity, stats, model) и считает суммарный `risk_score` в диапазоне `[0.0, 100.0]`.

### Статусы и их обработка

**`SKIPPED`** — сканер пропущен (нет входных данных или отключён). В скор не входит, фиксируется в `skipped_scanners`.

**`HAND_CHECK`** — сканер сигнализирует о необходимости ручной проверки:

| Категория сканера | Добавка к скору |
|---|---|
| `sanity`, `model` | +25.0 |
| `stats` | +8.0 |

**`FAILED`** — сканер нашёл проблему или завершился ошибкой:

| Условие | Действие |
|---|---|
| `reason = dataset_load_failed` или `model_load_failed` | `hard_block = True`, `score = max(score, 100.0)` |
| Категория `sanity`/`model` + явный `decision = BLOCK` | `hard_block = True`, `score = max(score, 90.0)` |
| Категория `sanity`/`model`, без явного BLOCK | +35.0 к скору |
| Категория `stats` | +12.0 к скору |
| Прочие | +20.0 к скору |

Stats-блок намеренно не вызывает hard_block в одиночку: каждый статистический детектор — улика, а не приговор. BLOCK по статистике возможен только при накоплении скора от множества детекторов.

### Финальный вердикт

```
hard_block = True          → BLOCK
score >= block_threshold   → BLOCK
score >= review_threshold  → REVIEW
иначе                      → ALLOW
```

---

## Итоговый объект `FinalDecision`

```python
{
  "decision":         "ALLOW" | "REVIEW" | "BLOCK",
  "risk_score":       float,          # суммарный скор [0.0, 100.0]
  "reasons":          List[str],      # человекочитаемые причины
  "category_scores":  {               # вклад каждой категории
    "sanity":  float,
    "stats":   float,
    "model":   float,
    "runtime": float,
  },
  "failed_scanners":  List[str],
  "review_scanners":  List[str],
  "skipped_scanners": List[str],
}
```

---

## Примеры

**Секрет в датасете:**
Sanity находит finding категории `secret` → always-block → Sanity возвращает BLOCK → агрегатор получает `FAILED` + `decision=BLOCK` → `hard_block=True` → **BLOCK**

**Несколько статистических аномалий:**
Stats-блок: 4 детектора вернули FAILED → 4 × 12.0 = 48.0 баллов. Если `review_threshold = 40` → **REVIEW**. Если ни один детектор не перешёл порог — **ALLOW**.

**Технический сбой при загрузке датасета:**
Сканер вернул `FAILED` с `reason=dataset_load_failed` → `hard_block=True` → **BLOCK** (пайплайн не может работать без данных).

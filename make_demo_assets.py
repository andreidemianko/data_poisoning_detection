"""
Создаёт РЕАЛИСТИЧНЫЕ демо-ассеты для проверки model-сканера:
  data/demo_poisoned.csv  — табличный датасет с подсаженным backdoor
  models/demo_mlp.pt      — обученный MLP (2 скрытых слоя), .pt со state_dict

Зачем отдельно от `init-demo`: встроенный init-demo делает одно-слойный
Linear(4,2) и 3 строки текста — model-сканер такую модель корректно ПРОПУСКАЕТ
(нет скрытого представления для Spectral/AC). Здесь — нормальный MLP, на котором
сканер реально отрабатывает и помечает датасет на ревью.

Запуск из корня проекта:
    python make_demo_assets.py
Потом:
    python -m src.cli scan --dataset data/demo_poisoned.csv --model models/demo_mlp.pt --mode only --only model
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
# Находим корень проекта (папку с requirements.txt / src/core), даже если скрипт
# лежит глубже (например, внутри src/scanners/model_scan).
for _candidate in [ROOT, *ROOT.parents]:
    if (_candidate / "requirements.txt").exists() or (_candidate / "src" / "core").is_dir():
        ROOT = _candidate
        break
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(0)
N, D = 4000, 8

# Чистый сигнал: 3 класса как argmax трёх линейных комбинаций признаков.
X = rng.normal(0, 1, (N, D)).astype("float32")
score = np.stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3], X[:, 4] + X[:, 5]], axis=1)
y = score.argmax(axis=1)

# Backdoor: на 10% строк ставим триггер-паттерн (f0=f1=4.0) и метим в класс 0.
trigger_idx = rng.choice(N, size=int(0.10 * N), replace=False)
X[trigger_idx, 0] = 4.0
X[trigger_idx, 1] = 4.0
y[trigger_idx] = 0

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(D)])
df["label"] = y                                   # колонка-метка (сканер ищет label/target/y/class)
df.to_csv(DATA / "demo_poisoned.csv", index=False)

# Обучаем небольшой MLP с 2 скрытыми слоями (ReLU) -> у него есть представление.
torch.manual_seed(0)
Xt = torch.from_numpy(X)
yt = torch.from_numpy(y).long()
model = nn.Sequential(
    nn.Linear(D, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 3),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model.train()
for _ in range(300):
    opt.zero_grad()
    loss = loss_fn(model(Xt), yt)
    loss.backward()
    opt.step()

torch.save(model.state_dict(), MODELS / "demo_mlp.pt")

# Короткий отчёт в консоль.
model.eval()
with torch.no_grad():
    pred = model(Xt).argmax(1).numpy()
acc = float((pred == y).mean())
asr = float((pred[trigger_idx] == 0).mean())     # доля триггер-строк, ушедших в целевой класс
print(f"created {DATA/'demo_poisoned.csv'}  ({N} строк, {D} признаков, 3 класса, 10% backdoor)")
print(f"created {MODELS/'demo_mlp.pt'}  (MLP 8-32-16-3, train acc={acc:.3f}, backdoor ASR={asr:.3f})")
print("\nтеперь запусти:")
print("  python -m src.cli scan --dataset data/demo_poisoned.csv --model models/demo_mlp.pt --mode only --only model")

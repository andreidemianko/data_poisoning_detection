"""
NLP model-level detectors: run Spectral / Activation Clustering / kNN-consensus on
the fine-tuned DistilBERT's embeddings, and RPP on the model itself (Gaussian noise
in the input embeddings). Ported from the benchmark (nlp_detectors.embed_bert and
rpp_nlp_detector.rpp_scores_bert).

Why on the *fine-tuned* model: it learned the trigger (trained on the poisoned
version), so backdoor rows cluster in its representation and stay stable under noise.

Requires torch + transformers (lazy-imported). NOT testable in the sandbox; logic
mirrors the validated benchmark code and is meant to run on the server with the
saved models. The four methods themselves (spectral/AC/kNN) are already validated.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

# как в бенчмарке (detect_cols)
TEXT_CANDS = ["text", "sentence", "instruction", "input", "tweet", "content"]
LABEL_CANDS = ["label", "labels", "sentiment", "output", "category", "class", "intent"]

_MODEL_CACHE: dict = {}   # model_dir -> (tokenizer, model, device)
_EMB_CACHE: dict = {}     # (model_dir, id(df)) -> (emb, y, label_col, text_cols)


def _progress(iterable, desc, total=None):
    """Прогресс-бар, если есть tqdm; иначе просто итерируем (видно, что не зависло)."""
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=False)
    except Exception:
        return iterable


# --------------------------------------------------------------------------- #
#  Resolve the model directory & columns
# --------------------------------------------------------------------------- #
def resolve_model_dir(model_path: str) -> str:
    """transformers грузит из ПАПКИ (config.json + tokenizer + model.safetensors).
    Если дали файл (.../model.safetensors) — берём его папку."""
    if model_path and os.path.isdir(model_path):
        return model_path
    return os.path.dirname(model_path) or model_path


def is_transformer_dir(model_dir: str) -> bool:
    return bool(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json"))


def find_text_label(df) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """(текстовые колонки, колонка-метка, причина-если-неуспех)."""
    from post_train_guard.detectors.common import text_columns

    low = {c.lower(): c for c in df.columns}
    text_cols = [low[k] for k in TEXT_CANDS if k in low] or text_columns(df)
    if not text_cols:
        return None, None, "no text columns"
    label_col = next((low[k] for k in LABEL_CANDS if k in low), None)
    if label_col is None:  # запасной вариант: низко-кардинальная не-текстовая колонка
        cap = max(50, int(0.1 * len(df)))
        for c in df.columns:
            if c in text_cols:
                continue
            nun = int(df[c].nunique(dropna=True))
            if 2 <= nun <= cap:
                label_col = c
                break
    if label_col is None:
        return text_cols, None, "no label column for per-class methods (Spectral/AC/kNN)"
    return text_cols, label_col, None


# --------------------------------------------------------------------------- #
#  Load fine-tuned classifier (lazy torch/transformers)
# --------------------------------------------------------------------------- #
def load_text_classifier(model_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_dir not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_dir, output_hidden_states=True)
        mdl.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)
        _MODEL_CACHE[model_dir] = (tok, mdl, dev)
    return _MODEL_CACHE[model_dir]


def _texts_of(df, text_cols: List[str]) -> List[str]:
    from post_train_guard.detectors.common import combined_text
    return [str(t) for t in combined_text(df, text_cols)]


# --------------------------------------------------------------------------- #
#  Embeddings (Spectral / AC / kNN) — кэш на время прогона
# --------------------------------------------------------------------------- #
def representation(model_path: str, df, max_len: int = 128, batch: int = 64):
    """(emb [n,H] mean-pooled последнего слоя, y int, label_col, text_cols).
    Считается один раз на (модель, датафрейм) и переиспользуется тремя адаптерами.
    Бросает ValueError с причиной (адаптер -> SKIPPED)."""
    import pandas as pd

    model_dir = resolve_model_dir(model_path)
    if not is_transformer_dir(model_dir):
        raise ValueError("model is not a HuggingFace transformer dir (no config.json)")
    text_cols, label_col, reason = find_text_label(df)
    if text_cols is None or label_col is None:
        raise ValueError(reason)

    key = (model_dir, id(df))
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]

    import torch
    tok, mdl, dev = load_text_classifier(model_dir)
    texts = _texts_of(df, text_cols)
    chunks = []
    n_batches = (len(texts) + batch - 1) // batch
    with torch.no_grad():
        for i in _progress(range(0, len(texts), batch), "BERT embed", total=n_batches):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            hs = mdl(**enc).hidden_states[-1]                # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(pooled.cpu().numpy())
    emb = np.vstack(chunks)
    y = pd.factorize(df[label_col].astype(str), sort=True)[0].astype(int)
    res = (emb, y, label_col, text_cols)
    _EMB_CACHE[key] = res
    return res


# --------------------------------------------------------------------------- #
#  RPP on the fine-tuned BERT — Gaussian noise in the INPUT embeddings (L-inf)
# --------------------------------------------------------------------------- #
def rpp_scores_bert(model_path: str, df, n_perturb: int = 8, sigma: float = 0.15,
                    max_len: int = 128, batch: int = 64) -> Tuple[np.ndarray, str, List[str]]:
    import torch

    model_dir = resolve_model_dir(model_path)
    if not is_transformer_dir(model_dir):
        raise ValueError("model is not a HuggingFace transformer dir (no config.json)")
    text_cols, label_col, reason = find_text_label(df)
    if text_cols is None:
        raise ValueError(reason or "no text columns")

    tok, mdl, dev = load_text_classifier(model_dir)
    emb_layer = mdl.get_input_embeddings()
    texts = _texts_of(df, text_cols)
    out = []
    n_batches = (len(texts) + batch - 1) // batch
    with torch.no_grad():
        for i in _progress(range(0, len(texts), batch), f"RPP | BERT x{n_perturb}", total=n_batches):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            ids, amask = enc["input_ids"], enc["attention_mask"]
            embeds = emb_layer(ids)
            p0 = torch.softmax(mdl(inputs_embeds=embeds, attention_mask=amask).logits, dim=-1)
            std = embeds.detach().std()
            m = amask.unsqueeze(-1).float()                  # шум только на реальных токенах
            pert = torch.zeros(embeds.shape[0], device=dev)
            for _ in range(n_perturb):
                noise = torch.randn_like(embeds) * (sigma * std) * m
                pk = torch.softmax(mdl(inputs_embeds=embeds + noise, attention_mask=amask).logits, dim=-1)
                pert += (pk - p0).abs().amax(dim=-1)         # L-inf, как в статье RPP
            out.append((pert / n_perturb).cpu().numpy())
    return -np.concatenate(out), (label_col or ""), text_cols  # стабильно -> подозрительно

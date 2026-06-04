"""
Minimal self-contained loaders so the engine can run standalone (like
dataset_guard/readers.py). The host adapter usually passes an already-loaded
DataFrame and (in the MODEL phase) a state_dict, so these are only fallbacks.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def read_dataset(path: str):
    """Читает табличный/текстовый датасет в DataFrame (csv/parquet/jsonl/json)."""
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def load_state_dict(path: str) -> Optional[Dict[str, Any]]:
    """Грузит state_dict обычной модели (.pt/.pth/.bin через torch, .safetensors).
    Возвращает None, если путь не годится — model-level детекторы тогда пропустятся."""
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".safetensors":
            from safetensors.torch import load_file
            return load_file(path)
        if ext in (".pt", ".pth", ".bin"):
            import torch
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    return None

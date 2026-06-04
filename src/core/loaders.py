from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    path: str


@dataclass
class ModelBundle:
    state_dict: Dict[str, Any]
    path: str
    metadata: Dict[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_under(base: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == base.name:
            candidate = base.parent / candidate
        else:
            candidate = base / candidate
    candidate = candidate.resolve()
    base = base.resolve()
    if base not in candidate.parents and candidate != base:
        raise ValueError(f"Path must be under {base}: {candidate}")
    return candidate


def resolve_dataset_path(raw_path: str) -> Path:
    candidate = Path(raw_path)

    if candidate.is_absolute():
        return candidate.resolve()

    return (project_root() / candidate).resolve()


def resolve_model_path(raw_path: str) -> Path:
    return _resolve_under(project_root() / "models", raw_path)


def load_dataset(raw_path: str) -> DatasetBundle:
    path = resolve_dataset_path(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    elif suffix == ".jsonl":
        data = pd.read_json(path, lines=True)
    elif suffix == ".json":
        data = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    return DatasetBundle(data=data, path=str(path))


def _state_dict_from_object(model_obj: Any) -> Dict[str, Any]:
    if hasattr(model_obj, "state_dict"):
        return model_obj.state_dict()
    if isinstance(model_obj, dict):
        if "state_dict" in model_obj and isinstance(model_obj["state_dict"], dict):
            return model_obj["state_dict"]
        return model_obj
    raise ValueError("Unsupported model object type")


def load_model(raw_path: str) -> ModelBundle:
    path = resolve_model_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Create one under models/ or run: python -m src.cli init-demo"
        )
    suffix = path.suffix.lower()
    metadata: Dict[str, Any] = {}
    if suffix == ".pt":
        model_obj = torch.load(path, map_location="cpu")
        state_dict = _state_dict_from_object(model_obj)
    elif suffix in {".safetensors", ".sft"}:
        state_dict = load_safetensors(str(path))
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")
    metadata["num_params"] = sum(int(t.numel()) for t in state_dict.values() if hasattr(t, "numel"))
    return ModelBundle(state_dict=state_dict, path=str(path), metadata=metadata)

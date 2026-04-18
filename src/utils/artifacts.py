import json
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = {"LSTM", "GRU"}


def get_model_type(config: dict[str, Any]) -> str:
    model_type = str(config["model_params"].get("model_type", "LSTM")).upper()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported values: {sorted(SUPPORTED_MODEL_TYPES)}"
        )
    return model_type


def get_models_dir(config: dict[str, Any]) -> Path:
    return Path(config["model"]["models_dir"])


def get_model_filename(model_type: str) -> str:
    return f"{model_type.lower()}_autoencoder.pth"


def get_model_path(config: dict[str, Any]) -> Path:
    model_type = get_model_type(config)
    return get_models_dir(config) / get_model_filename(model_type)


def get_threshold_path(config: dict[str, Any]) -> Path:
    return get_models_dir(config) / "threshold.json"


def load_threshold(config: dict[str, Any]) -> float:
    threshold_path = get_threshold_path(config)
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if "threshold" in payload:
                return float(payload["threshold"])

    return float(config["model_params"]["threshold"])

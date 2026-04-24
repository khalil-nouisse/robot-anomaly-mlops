import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

from src.inference.detect import run_inference
from src.training.train import train_model
from src.utils.core import load_config, setup_logger

logger = setup_logger()

CONFIG_PATH = Path("configs/config.yaml")


def _set_trial_seed(seed: int) -> None:
    """Set RNG seeds to reduce trial-to-trial variance.

    Returns:
        None: Applies seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_trial_config(base_config: dict, trial: optuna.trial.Trial, epochs: int | None) -> dict:
    """Create a trial-specific config by sampling hyperparameters with Optuna.

    Returns:
        dict: A deep-copied config dictionary updated with sampled model parameters.
    """
    config = copy.deepcopy(base_config)
    params = config["model_params"]

    # Focus search around the current best run neighborhood:
    # GRU, hidden_dim=24, n_layers=1, dropout=0.2, lr=0.001, batch_size=32.
    params["model_type"] = trial.suggest_categorical("model_type", ["GRU"])
    params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [22, 24, 26, 28, 32])
    params["n_layers"] = trial.suggest_categorical("n_layers", [1])
    params["dropout"] = trial.suggest_float("dropout", 0.10, 0.25, step=0.05)
    params["learning_rate"] = trial.suggest_float("learning_rate", 7e-4, 1.3e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32])
    if epochs is not None:
        params["epochs"] = int(epochs)

    return config


def _objective(trial: optuna.trial.Trial, base_config: dict, epochs: int | None) -> float:
    """Run one Optuna trial using existing training and detection pipelines.

    Returns:
        float: The AUROC score produced by `run_inference`, used as optimization target.
    """
    _set_trial_seed(42 + trial.number)
    trial_config = _build_trial_config(base_config, trial, epochs)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(trial_config, handle, sort_keys=False)

    train_model()
    metrics = run_inference(return_metrics=True)

    trial.set_user_attr("f1_score", metrics["f1_score"])
    trial.set_user_attr("precision", metrics["precision"])
    trial.set_user_attr("recall", metrics["recall"])
    trial.set_user_attr("threshold", metrics["calculated_threshold"])
    trial.set_user_attr("threshold_percentile", metrics["threshold_percentile"])
    return float(metrics["auroc_score"])


def run_tuning(
    trials: int,
    epochs: int | None,
    storage: str | None,
    study_name: str | None,
    warm_start_best: bool,
) -> None:
    """Execute an Optuna study and persist the best trial summary.

    Returns:
        None: Writes best trial details to `models/best_optuna_params.json`.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    original_config_text = CONFIG_PATH.read_text(encoding="utf-8")
    base_config = load_config(str(CONFIG_PATH))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    if warm_start_best:
        study.enqueue_trial(
            {
                "model_type": "GRU",
                "hidden_dim": 24,
                "n_layers": 1,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
            }
        )

    try:
        study.optimize(lambda trial: _objective(trial, base_config, epochs), n_trials=trials)
    finally:
        CONFIG_PATH.write_text(original_config_text, encoding="utf-8")

    best = study.best_trial
    summary = {
        "best_trial_number": best.number,
        "best_auroc_score": best.value,
        "best_params": best.params,
        "best_attrs": best.user_attrs,
    }

    models_dir = Path(base_config["model"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "best_optuna_params.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("==== OPTUNA BEST RESULT ====")
    logger.info(f"Best trial: {best.number}")
    logger.info(f"Best AUROC: {best.value:.6f}")
    logger.info(f"Best params: {best.params}")
    logger.info(f"Saved best summary to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for tuning execution.

    Returns:
        argparse.Namespace: Parsed CLI args for Optuna tuning.
    """
    parser = argparse.ArgumentParser(description="Tune RoboGuard hyperparameters with Optuna.")
    parser.add_argument("--trials", type=int, default=12, help="Number of Optuna trials.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Override epochs for every trial (optional).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="Optuna storage URL so studies can be resumed.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="roboguard_gru_local_refine",
        help="Study name used with the selected storage.",
    )
    parser.add_argument(
        "--no-warm-start-best",
        action="store_true",
        help="Disable enqueuing your known best GRU config as the first trial.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tuning(
        trials=args.trials,
        epochs=args.epochs,
        storage=args.storage,
        study_name=args.study_name,
        warm_start_best=not args.no_warm_start_best,
    )

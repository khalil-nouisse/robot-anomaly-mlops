import yaml
import logging
import torch
from pathlib import Path

def setup_logger(name="RoboGuard"):
    """Creates a standardized logger for the entire pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads the YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file missing at: {config_file}")
    
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def get_device():
    """Detects the best available hardware accelerator (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
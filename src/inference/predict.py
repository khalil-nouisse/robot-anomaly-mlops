import torch
import joblib
import numpy as np
from pathlib import Path
import logging

from src.models.autoencoder import LSTMAutoencoder
from src.utils.core import load_config, get_device
from src.utils.artifacts import get_model_path, get_model_type, load_threshold

logger = logging.getLogger("RoboGuard")

class RoboGuardPredictor:
    """Encapsulates all ML logic so the API remains completely separated from PyTorch."""
    
    def __init__(self):
        self.config = load_config()
        self.device = get_device()
        self.model = None
        self.scaler = None

        self.threshold = load_threshold(self.config)
        self.model_type = get_model_type(self.config)
        self.artifacts_loaded = False

    def load_artifacts(self):
        """Loads the weights and scaler from disk into memory."""
        models_dir = Path(self.config['model']['models_dir'])
        
        # Load Scaler
        scaler_path = models_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler artifact missing.")
        self.scaler = joblib.load(scaler_path)
        expected_features = int(self.config["model_params"]["n_features"])
        n_features_in = getattr(self.scaler, "n_features_in_", None)
        if n_features_in is not None and int(n_features_in) != expected_features:
            raise ValueError(
                f"Scaler features mismatch: expected {expected_features}, got {n_features_in}"
            )
        
        # Load Model
        params = self.config['model_params']
        model_type = self.model_type
        if model_type == 'GRU':
            from src.models.autoencoder import GRUAutoencoder
            self.model = GRUAutoencoder(
                n_features=params['n_features'],
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers']
            ).to(self.device)
        else:
            self.model = LSTMAutoencoder(
                n_features=params['n_features'],
                hidden_dim=params['hidden_dim'],
                n_layers=params['n_layers']
            ).to(self.device)
        
        model_path = get_model_path(self.config)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint missing at {model_path}")
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Weights in '{model_path.name}' do not match {model_type} architecture."
            ) from e

        self.model.eval()
        self.artifacts_loaded = True
        
        logger.info("Predictor artifacts loaded successfully into RAM.")

    def predict(self, raw_data: np.ndarray) -> tuple[bool, float]:
        """Handles the scaling, tensor math, and thresholding."""
        if not self.artifacts_loaded or self.model is None or self.scaler is None:
            raise RuntimeError("Predictor artifacts are not loaded.")
        # 1. Scale
        scaled_data = self.scaler.transform(raw_data)
        
        # 2. Convert to Tensor : scaled_data.shape = (250, 130) -> (batch of size=1, 250, 130)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            reconstruction = self.model(tensor_data)
            criterion = torch.nn.MSELoss()
            error_score = criterion(reconstruction, tensor_data).item()
            
        # 4. Threshold Logic
        is_anomaly = bool(error_score > self.threshold)
        
        return is_anomaly, error_score

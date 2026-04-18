import torch
import joblib
import numpy as np
from pathlib import Path
import logging

from src.models.autoencoder import LSTMAutoencoder
from src.utils.core import load_config, get_device

logger = logging.getLogger("RoboGuard")

class RoboGuardPredictor:
    """Encapsulates all ML logic so the API remains completely separated from PyTorch."""
    
    def __init__(self):
        self.config = load_config()
        self.device = get_device()
        self.model = None
        self.scaler = None

        self.threshold = self.config['model_params']['threshold']

    def load_artifacts(self):
        """Loads the weights and scaler from disk into memory."""
        models_dir = Path(self.config['model']['models_dir'])
        
        # Load Scaler
        scaler_path = models_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler artifact missing.")
        self.scaler = joblib.load(scaler_path)
        
        # Load Model
        params = self.config['model_params']
        model_type = params.get('model_type', 'LSTM')

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
        
        # Determine the expected file name (e.g. 'lstm_autoencoder.pth' or 'gru_autoencoder.pth')
        preferred_name = f"{model_type.lower()}_autoencoder.pth"
        model_path = models_dir / preferred_name
        
        # Fallback mechanism: if they asked for GRU but the only file is named 'lstm_...pth', use it
        if model_type == 'GRU' and not model_path.exists():
            fallback_path = models_dir / "lstm_autoencoder.pth"
            if fallback_path.exists():
                model_path = fallback_path
                
        if model_path.exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            except RuntimeError as e:
                logger.error(f"SHAPE MISMATCH: The weights in '{model_path.name}' do not match the {model_type} architecture. The API will start, but the AI is untrained! Error: {e}")
        else:
            logger.error(f"CHECKPOINT MISSING: Could not find {model_path}. The API will start, but the AI is untrained!")

        self.model.eval()
        
        logger.info("Predictor artifacts loaded successfully into RAM.")

    def predict(self, raw_data: np.ndarray) -> tuple[bool, float]:
        """Handles the scaling, tensor math, and thresholding."""
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
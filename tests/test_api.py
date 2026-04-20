import pytest
from fastapi.testclient import TestClient
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# Import your FastAPI app and config loader
from api.main import app
from src.utils.core import load_config
from src.models.autoencoder import LSTMAutoencoder, GRUAutoencoder
from src.utils.artifacts import get_model_path, get_model_type

# 1. Initialize the TestClient
# Note: Using 'with TestClient(app)' triggers your @asynccontextmanager lifespan 
# so the model and scaler load perfectly into RAM for the tests!
@pytest.fixture(scope="module")
def client():
    config = load_config()
    model_path = get_model_path(config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    params = config["model_params"]
    model_type = get_model_type(config)
    n_features = int(params["n_features"])

    # Make API tests independent from external/downstream downloaded artifacts.
    scaler = StandardScaler().fit(np.random.randn(256, n_features))
    joblib.dump(scaler, model_path.parent / "feature_scaler.pkl")

    if model_type == "GRU":
        model = GRUAutoencoder(
            n_features=n_features,
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            dropout=params.get("dropout", 0.0),
        )
    else:
        model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            dropout=params.get("dropout", 0.0),
        )
    torch.save(model.state_dict(), model_path)
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def config():
    return load_config()

def test_predict_success(client, config):
    """Test the Happy Path: Valid shape payload."""
    seq_len = config['model']['fixed_length']
    n_features = config['model_params']['n_features']
    
    # Generate a fake sequence of the exact correct shape
    fake_sequence = np.zeros((seq_len, n_features)).tolist()
    
    payload = {"sequence": fake_sequence}
    
    # Send the request
    response = client.post("/predict", json=payload)
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "threshold_used" in data
    assert "status" in data

def test_predict_invalid_shape(client, config):
    """Test the Error Path: Missing time steps (e.g., sensor failure)."""
    seq_len = config['model']['fixed_length']
    n_features = config['model_params']['n_features']
    
    # Create a sequence that is 50 steps too short
    bad_seq_len = seq_len - 50 
    bad_sequence = np.zeros((bad_seq_len, n_features)).tolist()
    
    payload = {"sequence": bad_sequence}
    
    response = client.post("/predict", json=payload)
    
    # Assertions
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Invalid shape" in data["detail"]

def test_predict_invalid_features(client, config):
    """Test the Error Path: Missing sensor columns."""
    seq_len = config['model']['fixed_length']
    n_features = config['model_params']['n_features']
    
    # Create a sequence missing one entire feature column
    bad_features = n_features - 1
    bad_sequence = np.zeros((seq_len, bad_features)).tolist()
    
    payload = {"sequence": bad_sequence}
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 400

def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_readyz(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model_type"] in {"LSTM", "GRU"}

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.inference.predict import RoboGuardPredictor
from src.models.autoencoder import GRUAutoencoder, LSTMAutoencoder


def _make_config(models_dir, model_type="LSTM", n_features=3):
    return {
        "data": {"processed_dir": "data/processed"},
        "model": {"models_dir": str(models_dir), "fixed_length": 4},
        "model_params": {
            "model_type": model_type,
            "n_features": n_features,
            "hidden_dim": 2,
            "n_layers": 1,
            "dropout": 0.0,
            "threshold": 0.1,
        },
    }


def test_load_artifacts_fails_when_scaler_missing(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path)
    monkeypatch.setattr("src.inference.predict.load_config", lambda: cfg)
    predictor = RoboGuardPredictor()
    with pytest.raises(FileNotFoundError, match="Scaler artifact missing"):
        predictor.load_artifacts()


def test_load_artifacts_fails_when_weights_mismatch(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path, model_type="GRU")
    models_dir = tmp_path

    scaler = StandardScaler().fit(np.random.randn(20, 3))
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    lstm = LSTMAutoencoder(n_features=3, hidden_dim=2, n_layers=1, dropout=0.0)
    torch.save(lstm.state_dict(), models_dir / "gru_autoencoder.pth")

    monkeypatch.setattr("src.inference.predict.load_config", lambda: cfg)
    predictor = RoboGuardPredictor()
    with pytest.raises(RuntimeError, match="do not match GRU architecture"):
        predictor.load_artifacts()


def test_predict_uses_threshold_artifact(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path, model_type="GRU")
    models_dir = tmp_path

    scaler = StandardScaler().fit(np.random.randn(50, 3))
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")

    gru = GRUAutoencoder(n_features=3, hidden_dim=2, n_layers=1, dropout=0.0)
    torch.save(gru.state_dict(), models_dir / "gru_autoencoder.pth")

    (models_dir / "threshold.json").write_text(
        '{"model_type":"GRU","threshold":0.00001,"threshold_percentile":95}',
        encoding="utf-8",
    )

    monkeypatch.setattr("src.inference.predict.load_config", lambda: cfg)
    predictor = RoboGuardPredictor()
    predictor.load_artifacts()

    is_anomaly, score = predictor.predict(np.random.randn(4, 3))
    assert isinstance(is_anomaly, bool)
    assert isinstance(score, float)
    assert predictor.threshold == pytest.approx(0.00001)

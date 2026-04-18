import pytest

from src.utils.artifacts import (
    get_model_filename,
    get_model_path,
    get_model_type,
    load_threshold,
)


def _config(tmp_path, model_type="LSTM"):
    return {
        "model": {"models_dir": str(tmp_path)},
        "model_params": {"model_type": model_type, "threshold": 1.23},
    }


def test_model_filename_resolution():
    assert get_model_filename("LSTM") == "lstm_autoencoder.pth"
    assert get_model_filename("GRU") == "gru_autoencoder.pth"


def test_model_path_from_config(tmp_path):
    cfg = _config(tmp_path, model_type="GRU")
    assert str(get_model_path(cfg)).endswith("gru_autoencoder.pth")


def test_unsupported_model_type_raises(tmp_path):
    cfg = _config(tmp_path, model_type="CNN")
    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_model_type(cfg)


def test_threshold_falls_back_to_config(tmp_path):
    cfg = _config(tmp_path)
    assert load_threshold(cfg) == pytest.approx(1.23)


def test_threshold_prefers_artifact(tmp_path):
    cfg = _config(tmp_path)
    (tmp_path / "threshold.json").write_text('{"threshold": 0.42}', encoding="utf-8")
    assert load_threshold(cfg) == pytest.approx(0.42)

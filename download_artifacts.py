import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from src.utils.core import load_config
from src.utils.artifacts import get_model_filename, get_model_type

def download_latest_production_artifacts():
    print("Connecting to Cloud MLflow Registry...")
    
    # Set the URI from the GitHub Secret
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    client = MlflowClient()
    
    experiment_name = "Voraus_Robotic_Anomaly_Detection"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found in the cloud!")

    # Find the most recent successful run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No training runs found!")
        
    latest_run_id = runs[0].info.run_id
    print(f"Downloading artifacts from Run ID: {latest_run_id}")
    
    # Ensure the local models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    config = load_config()
    model_type = get_model_type(config)
    model_filename = get_model_filename(model_type)
    
    # 1. Download the Scaler
    print("Downloading Scaler...")
    local_scaler_path = client.download_artifacts(latest_run_id, "preprocessing/feature_scaler.pkl")
    shutil.move(local_scaler_path, "models/feature_scaler.pkl")
    
    # 2. Download the PyTorch Model
    print("Downloading PyTorch Model...")
    local_model_path = client.download_artifacts(latest_run_id, f"model/{model_filename}")
    shutil.move(local_model_path, f"models/{model_filename}")

    # 3. Download threshold artifact when available
    try:
        print("Downloading Threshold Artifact...")
        local_threshold_path = client.download_artifacts(latest_run_id, "threshold/threshold.json")
        shutil.move(local_threshold_path, "models/threshold.json")
    except Exception:
        print("Threshold artifact not found in this run. Falling back to config threshold at runtime.")
    
    print("All artifacts successfully downloaded from the cloud!")

if __name__ == "__main__":
    download_latest_production_artifacts()

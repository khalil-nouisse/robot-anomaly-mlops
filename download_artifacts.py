import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

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
    
    # 1. Download the Scaler
    print("Downloading Scaler...")
    local_scaler_path = client.download_artifacts(latest_run_id, "preprocessing/feature_scaler.pkl")
    shutil.move(local_scaler_path, "models/feature_scaler.pkl")
    
    # 2. Download the entire PyTorch Model folder
    print("Downloading PyTorch Model directory...")
    local_model_dir = client.download_artifacts(latest_run_id, "model")
    
    # 3. Recursively search inside the folder for the .pth weights file
    pth_files = list(Path(local_model_dir).rglob("*.pth"))
    
    if not pth_files:
        raise FileNotFoundError("Could not find a .pth file inside the MLflow model folder!")
        
    # Move the found weights file to the exact spot our API expects
    shutil.move(str(pth_files[0]), "models/lstm_autoencoder.pth")
    
    print("All artifacts successfully downloaded from the cloud!")

if __name__ == "__main__":
    download_latest_production_artifacts()
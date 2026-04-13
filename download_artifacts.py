import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

def scan_and_find_weights(client, run_id, path="", indent=""):
    """Recursively prints the cloud bucket structure AND hunts for the model."""
    found_file = None
    artifacts = client.list_artifacts(run_id, path)
    
    for file_info in artifacts:
        # Print the file structure so we can see it in GitHub logs!
        print(f"{indent} |- {file_info.path}")
        
        if file_info.is_dir:
            # Dig deeper into the folder
            res = scan_and_find_weights(client, run_id, file_info.path, indent + "  ")
            if res and not found_file:
                found_file = res
        else:
            # Broaden the search: Look for .pth, .pt, or .pkl inside the model directory
            if "model/data/" in file_info.path and file_info.path.endswith((".pth", ".pt", ".pkl")):
                found_file = file_info.path
                
    return found_file

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
    
    # 2. Map the bucket and find the PyTorch model
    print("Scanning cloud bucket contents:")
    pth_cloud_path = scan_and_find_weights(client, latest_run_id)
    
    if not pth_cloud_path:
        raise FileNotFoundError("Could not find the model weights! Check the printed file tree above.")
        
    print(f"\nTarget acquired! Found weights at: {pth_cloud_path}. Downloading...")
    local_model_path = client.download_artifacts(latest_run_id, pth_cloud_path)
    shutil.move(local_model_path, "models/lstm_autoencoder.pth")
    
    print("All artifacts successfully downloaded from the cloud!")

if __name__ == "__main__":
    download_latest_production_artifacts()
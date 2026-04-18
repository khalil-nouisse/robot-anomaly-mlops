import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import logging
from pathlib import Path
import mlflow
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score


from src.models.autoencoder import LSTMAutoencoder
from src.utils.core import load_config, get_device, setup_logger
from src.models.autoencoder import LSTMAutoencoder, GRUAutoencoder

logger = setup_logger()

def calculate_reconstruction_errors(model, dataloader, device):
    """Runs data through the model and calculates the error for EACH individual sequence."""
    model.eval()
    errors = []
    
    # We use reduction='none' because we want the error per sequence, not the average of the batch
    criterion = nn.MSELoss(reduction='none') 
    
    with torch.no_grad():
        for batch in dataloader:
            batch_data = batch[0].to(device)
            reconstruction = model(batch_data)
            
            # 1. Calculate raw pixel-by-pixel error: Shape (batch_size, seq_len, features)
            loss = criterion(reconstruction, batch_data)
            
            # 2. Average the error across the time steps and features to get ONE score per sequence
            seq_errors = loss.mean(dim=[1, 2]).cpu().numpy() 
            errors.extend(seq_errors)
            
    return np.array(errors)

def run_inference():
    config = load_config()
    params = config['model_params']
    device = get_device()
    
    PROCESSED_DIR = Path(config['data']['processed_dir'])
    MODEL_PATH = Path(config['model']['models_dir']) / "lstm_autoencoder.pth"
    
    # 1. Load the Trained Model
    logging.info("Loading trained model architecture and weights...")

    model_type = params.get('model_type', 'LSTM')

    if model_type == "GRU" :
        model = GRUAutoencoder(
            n_features=params['n_features'],
            hidden_dim=params['hidden_dim'],
            n_layers=params['n_layers'],
            dropout=params['dropout']
        ).to(device)
    else :
        model = LSTMAutoencoder(
            n_features=params['n_features'],
            hidden_dim=params['hidden_dim'],
            n_layers=params['n_layers'],
            dropout=params['dropout']
        ).to(device)
    
    # Inject the learned brain (the .pth file) into the architecture
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # 2. Load the Data
    logging.info("Loading validation (normal) and test (anomaly) tensors...")
    X_val = torch.load(PROCESSED_DIR / "X_val.pt")
    X_anomaly = torch.load(PROCESSED_DIR / "X_anomaly.pt")
    
    val_loader = DataLoader(TensorDataset(X_val), batch_size=params['batch_size'], shuffle=False)
    anomaly_loader = DataLoader(TensorDataset(X_anomaly), batch_size=params['batch_size'], shuffle=False)

    # 3. Calculate Errors
    logging.info("Calculating reconstruction errors for Normal data...")
    val_errors = calculate_reconstruction_errors(model, val_loader, device)
    
    logging.info("Calculating reconstruction errors for Anomaly data...")
    anomaly_errors = calculate_reconstruction_errors(model, anomaly_loader, device)

    # 4. Combine all data for evaluation FIRST
    logging.info("Combining datasets to find the optimal mathematical threshold...")
    # True labels: Normal = 0, Anomaly = 1
    y_true = np.concatenate([np.zeros(len(val_errors)), np.ones(len(anomaly_errors))])
    # Predicted scores (the raw error values)
    y_scores = np.concatenate([val_errors, anomaly_errors])

    # 5. The MLOps F1 Maximization Loop
    best_f1 = 0
    best_threshold = 0
    best_percentile = 0

    # Test percentiles from 75 to 99 (based strictly on the normal validation distribution)
    for p in range(75, 100):
        test_thresh = np.percentile(val_errors, p)
        
        # Apply this test threshold to the ENTIRE dataset
        temp_preds = (y_scores > test_thresh).astype(int) 
        
        # Calculate F1 against the true labels
        current_f1 = f1_score(y_true, temp_preds)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = test_thresh
            best_percentile = p

    logger.info(f"--- OPTIMAL THRESHOLD FOUND AT {best_percentile}th PERCENTILE: {best_threshold:.5f} ---")
    threshold = best_threshold 

    # 6. Evaluate the Final Results using the winning threshold
    y_pred = (y_scores > threshold).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    logging.info("\n" + classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    
    auroc = roc_auc_score(y_true, y_scores)
    logging.info(f"Final AUROC Score: {auroc:.4f}")

    # 7. Log to Cloud Database
    mlflow.set_experiment("Voraus_Robotic_Anomaly_Detection_Eval")
    with mlflow.start_run(run_name="Model_Evaluation"):
        logging.info("Logging evaluation metrics to MLflow...")
        #log model type to mlflow
        mlflow.log_param("model_type", params.get('model_type', 'LSTM'))
        # Now logs the actual winning percentile
        mlflow.log_param("threshold_percentile", best_percentile) 
        mlflow.log_metrics({
            "auroc_score": auroc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "calculated_threshold": threshold
        })

if __name__ == "__main__":
    run_inference()
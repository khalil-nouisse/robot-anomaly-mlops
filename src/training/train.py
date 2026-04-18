import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import time
import logging
from pathlib import Path
import mlflow

# Import our custom architecture
from src.models.autoencoder import LSTMAutoencoder
from src.utils.core import load_config, get_device, setup_logger
from src.utils.core import track_experiment
from src.models.autoencoder import LSTMAutoencoder, GRUAutoencoder

logger = setup_logger()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



@track_experiment(experiment_name="Voraus_Robotic_Anomaly_Detection")
def train_model():
    config = load_config()
    
    # Setup Paths & Params
    PROCESSED_DIR = Path(config['data']['processed_dir'])
    MODELS_DIR = Path(config['model']['models_dir'])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    params = config['model_params']
    device = get_device()
    logging.info(f"Using device: {device}")


    #log the model type to mlflow
    mlflow.log_param("model_type", params.get('model_type', 'LSTM'))
    #log the parameter models to mlflow
    mlflow.log_params(params)

    # Load Data
    logging.info("Loading tensors...")
    try:
        X_train = torch.load(PROCESSED_DIR / "X_train.pt")
        X_val = torch.load(PROCESSED_DIR / "X_val.pt")
    except FileNotFoundError:
        logging.error("Tensors not found. Did you run src.features.build_features?")
        return

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=params['batch_size'], shuffle=False)

    
    # Initialize Model, Loss, and Optimizer
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
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 5. PyTorch Training Loop

    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    logging.info("Starting Training Loop...")

    for epoch in range(params['epochs']):

        epoch_start_time = time.time()

        model.train() # This line turns Dropout On (20% of neurons to zero)
        train_loss = 0.0
        
        for batch in train_loader:
            batch_data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch_data)
            
            loss = criterion(reconstruction, batch_data)
            loss.backward()     #Backward pass - computing gradients
            optimizer.step()    #updates the parameters
            
            train_loss += loss.item() * batch_data.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Step
        model.eval()    #turns Dropout off , now the model uses 100% of its neurons
        val_loss = 0.0
        with torch.no_grad():   #tells pytorch : do not calculate any calculus derivatives (only testing)
            for batch in val_loader:
                batch_data = batch[0].to(device)
                reconstruction = model(batch_data)
                loss = criterion(reconstruction, batch_data)
                val_loss += loss.item() * batch_data.size(0)
        
        val_loss /= len(val_loader.dataset)

        epoch_end_time = time.time() 
        epoch_duration = epoch_end_time - epoch_start_time

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_duration_sec": epoch_duration
        }, step=epoch)

        # Print progress to the terminal
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch [{epoch+1}/{params['epochs']}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"--- Early Stopping Triggered at Epoch {epoch+1}! Loss has stabilized. ---")
            break # Break out of the epochs!

    # 6. Save the final model locally
    logging.info("Training complete. Saving model weights...")
    local_model_path = MODELS_DIR / "lstm_autoencoder.pth"
    torch.save(model.state_dict(), local_model_path)

    logging.info(f"Model saved successfully to {local_model_path}")

    # --- dagshub CLOUD REGISTRY ---
    # Log the raw PyTorch weights DIRECTLY to the cloud (The Bulletproof Way)
    mlflow.log_artifact(str(local_model_path), artifact_path="model")

    #  Log the Scaler so the cloud has a backup of it
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    if scaler_path.exists():
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        
    logger.info("Artifacts successfully securely pushed to Cloud MLflow Registry.")

if __name__ == "__main__":
    train_model()






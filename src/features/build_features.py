import pandas as pd
import numpy as np
import torch
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data.ingest import load_data
from src.utils.core import load_config, setup_logger
from src.models.autoencoder import LSTMAutoencoder

import yaml
from pathlib import Path


logger = setup_logger()
config = load_config()

# Extract variables from the config dictionary
RAW_DATA_PATH = Path(config['data']['raw_path'])
PROCESSED_DIR = Path(config['data']['processed_dir'])
MODELS_DIR = Path(config['model']['models_dir'])
FIXED_LENGTH = config['model']['fixed_length']

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts float64 to float32 to cut RAM usage in half."""

    logging.info("Downcasting float64 columns to float32...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def build_windows(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Splits samples into Train, Val, and Anomaly.
    Fits scaler ONLY on Train data (preventing leakage).
    Groups by 'sample', scales features, and creates fixed-length windows.
    """
    logging.info("Identifying normal and anomalous sequences...")
    
    # Figure out which 'samples' have anomalies and which are perfectly normal
    sample_stats = df.groupby('sample')['anomaly'].any()
    normal_sample_ids = sample_stats[~sample_stats].index.tolist()
    anomaly_sample_ids = sample_stats[sample_stats].index.tolist()

    # Split normal samples into Train and Validation at the SEQUENCE level
    train_ids, val_ids = train_test_split(normal_sample_ids, test_size=0.2, random_state=42)

    logging.info("Fitting StandardScaler on training sequences only...")
    scaler = StandardScaler()
    
    # Fit ONLY on the rows belonging to the training sequences
    train_mask = df['sample'].isin(train_ids)
    scaler.fit(df.loc[train_mask, feature_cols])
    
    # transform the entire dataset using the Train-fitted scaler
    df[feature_cols] = scaler.transform(df[feature_cols])

    # Save the scaler for inference
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    logging.info("Scaler saved to models/feature_scaler.pkl")

    X_train = []    #trainning windows
    X_val = []      #validation windows
    X_anomaly = []  #testing windows

    logging.info(f"Grouping sequences by 'sample' into {FIXED_LENGTH}-step windows...")
    grouped = df.groupby('sample')
    
    # Build windows and route them to the correct lists based on their sample ID
    for sample_id, group in grouped:
        sequence = group[feature_cols].values
        
        # Pad or Truncate
        seq_length = sequence.shape[0]
        if seq_length >= FIXED_LENGTH:
            processed_seq = sequence[:FIXED_LENGTH, :]
        else:
            padding = np.zeros((FIXED_LENGTH - seq_length, sequence.shape[1]), dtype=np.float32)
            processed_seq = np.vstack((sequence, padding))
            
        # Distribute into the correct bucket
        if sample_id in train_ids:
            X_train.append(processed_seq)
        elif sample_id in val_ids:
            X_val.append(processed_seq)
        else: 
            X_anomaly.append(processed_seq)

    return np.array(X_train), np.array(X_val), np.array(X_anomaly)



if __name__ == "__main__":

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load Data (Using our modular pipeline component!)
        logging.info("Loading raw dataset...")
        df = load_data(RAW_DATA_PATH)
        
        # Optimize Memory
        df = optimize_memory(df)
        
        # Identify physical signals (Ignore metadata)
        ignore_cols = ['time', 'sample', 'anomaly', 'category', 'setting', 'action', 'active']
        feature_cols = [col for col in df.columns if col not in ignore_cols]
        logging.info(f"Identified {len(feature_cols)} physical signal columns for training.")

        # Build Windows
        X_train, X_val, X_anomaly = build_windows(df, feature_cols)
        
        # Convert to PyTorch Tensors and Save
        logging.info("Converting arrays to PyTorch tensors...")
        torch.save(torch.tensor(X_train, dtype=torch.float32), PROCESSED_DIR / "X_train.pt")
        torch.save(torch.tensor(X_val, dtype=torch.float32), PROCESSED_DIR / "X_val.pt")
        torch.save(torch.tensor(X_anomaly, dtype=torch.float32), PROCESSED_DIR / "X_anomaly.pt")
        
        logging.info("--- Pipeline Success ---")
        logging.info(f"data shape (batch (number of windows) , equence length , features per timestep)")
        logging.info(f"Train Tensors Saved: {X_train.shape}")
        logging.info(f"Validation Tensors Saved: {X_val.shape}")
        logging.info(f"Anomaly Tensors Saved: {X_anomaly.shape}")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
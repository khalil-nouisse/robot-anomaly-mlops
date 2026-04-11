RoboGuard-AI

An End-to-End Deep Learning & MLOps System for Multivariate Time-Series Anomaly Detection in Robotic Systems

⸻

Overview

RoboGuard-AI is a complete Deep Learning + MLOps system designed and implemented end-to-end to detect anomalies in robotic operations using multivariate time-series data.

This project covers both:
	•	 Model Development (LSTM Autoencoder for anomaly detection)
	•	 MLOps Engineering (experiment tracking, reproducibility, deployment pipeline)

It demonstrates how to move from a research idea to a production-ready AI system. The project focuses on identifying faults such as:
	•	Axis wear and mechanical degradation
	•	Gripping failures in pick-and-place operations
	•	Sensor inconsistencies and abnormal patterns

It combines deep learning (LSTM Autoencoders) with an MLOps pipeline to enable scalable, production-ready monitoring.

⸻

 Key Features

Deep Learning
	•	 Real-time anomaly detection using LSTM Autoencoders
	•	 Multivariate time-series modeling
	•	 Model training, validation, and evaluation

MLOps
	•	 End-to-end ML pipeline (data → training → inference)
	•	 Experiment tracking (MLflow)
	•	 Reproducible workflows
	•	 Deployment-ready architecture (API / monitoring)
	•	 Performance monitoring & metrics

⸻

System Architecture

This project is structured as a full ML lifecycle pipeline:

[Data Engineering] → [Model Development] → [Experiment Tracking] → [Deployment] → [Monitoring]

Detailed Flow

Data Sources (Sensors / Logs)
        ↓
Data Preprocessing (Cleaning, Scaling, Windowing)
        ↓
Model Training (LSTM Autoencoder)
        ↓
Anomaly Scoring (Reconstruction Error)
        ↓
Monitoring & Alerts
        ↓
Deployment (API / Dashboard)


⸻

Tech Stack
	•	Languages: Python
	•	ML/DL: PyTorch, NumPy, Pandas
	•	Visualization: Matplotlib, Seaborn
	•	MLOps: MLflow (experiment tracking), Docker (optional)
	•	Backend (optional): FastAPI

⸻

Getting Started

1. Clone the repository

git clone https://github.com/your-username/roboguard-ai.git
cd roboguard-ai

2. Install dependencies

pip install -r requirements.txt

3. Run training

python train.py

4. Run anomaly detection

python detect.py


⸻

Example Use Case

The system analyzes time-series data from robotic joints and sensors. When abnormal behavior occurs (e.g., increased vibration or irregular motion), the model detects it using reconstruction error thresholds.

⸻

Project Structure

roboguard-ai/
│
├── .github/workflows/  # CI/CD pipelines (GitHub Actions for linting/testing)
├── configs/            # YAML files (Hyperparameters, DB connections, API settings)
├── data/               # Raw & processed datasets (Must be in .gitignore!)
├── docker/             # Dockerfiles (e.g., Dockerfile.api, Dockerfile.train)
├── models/             # Local model artifacts (weights, scalers) before MLflow
│
├── src/
│   ├── data/           # Data ingestion & validation scripts
│   ├── features/       # Preprocessing & time-series windowing
│   ├── models/         # Deep Learning architectures (PyTorch)
│   ├── training/       # Training loops & MLflow tracking integration
│   └── inference/      # Detection logic & anomaly scoring
│
├── api/                # FastAPI service (Mandatory for serving!)
├── tests/              # Unit tests for preprocessing logic and API endpoints
├── notebooks/          # Experiments & EDA (Strictly scratchpads)
│
├── .gitignore          # Crucial for ignoring the 1.1GB dataset and __pycache__
├── requirements.txt    # Or pyproject.toml (if using Poetry)
└── README.md


⸻

Metrics
	•	Reconstruction Error (MSE)
	•	Precision / Recall for anomaly detection
	•	ROC-AUC


⸻

Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

⸻

👤 Author

Khalil Ait Nouisse
Software Engineering Student | ML & AI Enthusiast
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import numpy as np

# Import our new clean Predictor class and Pydantic schemas
from src.inference.predict import RoboGuardPredictor
from src.utils.core import load_config
from api.schemas import RobotSequence, AnomalyResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter


config = load_config()

# Initialize the predictor (but don't load artifacts yet)
predictor = RoboGuardPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Triggered once when Uvicorn starts."""
    try:
        predictor.load_artifacts()
        yield  # <-- CRITICAL: Hands control over to the web server
    except Exception as e:
        raise RuntimeError(f"Failed to load AI models: {e}")

app = FastAPI(
    title="RoboGuard AI Inference API",
    lifespan=lifespan,
    description="Real-time Anomaly Detection for Pick-and-Place Robots",
    version="1.0.0"
)
anomaly_counter = Counter("roboguard_anomalies_total", "Total anomalies")
# Instrument the app to expose /metrics
Instrumentator().instrument(app).expose(app)

@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(payload: RobotSequence):
    """The 'Thin Router': Accepts HTTP data, passes it to the Predictor, returns JSON."""
    
    # 1. Extract data
    raw_data = np.array(payload.sequence)
    
    # 2. Validate Shape
    expected_steps = config['model']['fixed_length']
    expected_features = config['model_params']['n_features']
    
    if raw_data.shape != (expected_steps, expected_features):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid shape. Expected ({expected_steps}, {expected_features}), got {raw_data.shape}"
        )

    # 3. Delegate to the AI Engine
    try:
        is_anomaly, error_score = predictor.predict(raw_data)
        
        # Increment our custom Prometheus metric if an anomaly is found!
        if is_anomaly:
            anomaly_counter.inc(1)
        
        # 4. Return HTTP Response
        return AnomalyResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(error_score, 5),
            threshold_used=predictor.threshold,
            status="DANGER: Maintenance Required" if is_anomaly else "Operational"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Prints the exact line of the crash to your terminal
        raise HTTPException(status_code=500, detail=f"CRASH: {str(e)}") # Sends the real error to Pytest
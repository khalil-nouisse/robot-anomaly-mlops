from pydantic import BaseModel, Field
from typing import List

class RobotSequence(BaseModel):
    """
    Validates the incoming JSON payload.
    Expects a 2D array: 250 time steps, each containing 130 sensor readings.
    """
    sequence: List[List[float]] = Field(
        ..., 
        description="A 2D array of shape [250, 130] representing a single pick-and-place movement."
    )

class AnomalyResponse(BaseModel):
    """
    The structured response sent back to the robot.
    """
    is_anomaly: bool
    anomaly_score: float
    threshold_used: float
    status: str
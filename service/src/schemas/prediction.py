from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PredictionSummary(BaseModel):
    id: int
    terminal_id: int
    terminal_name: str
    confidence_score: float
    distance_to_terminal_km: float
    eta_hours: Optional[float]
    status: str


class PredictionDetail(PredictionSummary):
    vessel_id: str
    vessel_name: str
    predicted_arrival: Optional[datetime]
    proximity_score: float
    speed_score: float
    heading_score: float
    historical_similarity_score: float
    ai_confidence_adjustment: float
    ai_reasoning: str
    prediction_time: datetime

    class Config:
        from_attributes = True


class AnalyzeRequest(BaseModel):
    vessel_id: str


class ConfirmOutcomeRequest(BaseModel):
    actual_arrival_time: Optional[datetime] = None
    status: str = 'confirmed'  # or 'incorrect'

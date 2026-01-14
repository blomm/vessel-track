from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class VesselBase(BaseModel):
    name: str
    current_lat: float
    current_lon: float
    heading: Optional[float] = None
    speed: Optional[float] = None
    vessel_type: str = 'lng_tanker'


class VesselCreate(VesselBase):
    id: str
    mmsi: Optional[str] = None
    imo: Optional[str] = None


class VesselUpdate(BaseModel):
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
    status: Optional[str] = None


class VesselResponse(VesselBase):
    id: str
    mmsi: Optional[str]
    imo: Optional[str]
    status: Optional[str]
    last_updated: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class VesselWithPredictions(VesselResponse):
    active_predictions: List['PredictionSummary'] = []

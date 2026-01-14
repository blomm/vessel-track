from pydantic import BaseModel
from typing import Optional


class TerminalBase(BaseModel):
    name: str
    code: str
    lat: float
    lon: float
    country: str
    region: str
    terminal_type: str
    capacity_bcm_year: Optional[float] = None
    approach_zone_radius_km: float = 50.0


class TerminalCreate(TerminalBase):
    pass


class TerminalResponse(TerminalBase):
    id: int

    class Config:
        from_attributes = True

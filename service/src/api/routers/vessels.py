from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from src.database.connection import get_db
from src.database.models import Vessel, Prediction
from src.schemas.vessel import VesselResponse, VesselCreate, VesselUpdate
from src.schemas.prediction import PredictionSummary

router = APIRouter()


@router.get("", response_model=List[VesselResponse])
async def list_vessels(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """List all vessels with pagination"""
    result = await db.execute(
        select(Vessel).offset(skip).limit(limit)
    )
    vessels = result.scalars().all()
    return vessels


@router.get("/{vessel_id}")
async def get_vessel(
    vessel_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get specific vessel with active predictions"""
    result = await db.execute(
        select(Vessel).where(Vessel.id == vessel_id)
    )
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")

    # Get active predictions
    pred_result = await db.execute(
        select(Prediction).where(
            Prediction.vessel_id == vessel_id,
            Prediction.status == 'active'
        )
    )
    predictions = pred_result.scalars().all()

    # Build response
    vessel_dict = VesselResponse.model_validate(vessel).model_dump()
    vessel_dict['active_predictions'] = [
        PredictionSummary(
            id=p.id,
            terminal_id=p.terminal_id,
            terminal_name=p.terminal.name,
            confidence_score=p.confidence_score,
            distance_to_terminal_km=p.distance_to_terminal_km,
            eta_hours=p.eta_hours,
            status=p.status
        ) for p in predictions
    ]

    return vessel_dict


@router.post("", response_model=VesselResponse)
async def create_vessel(
    vessel: VesselCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new vessel"""
    db_vessel = Vessel(**vessel.model_dump())
    db.add(db_vessel)
    await db.commit()
    await db.refresh(db_vessel)
    return db_vessel


@router.put("/{vessel_id}", response_model=VesselResponse)
async def update_vessel(
    vessel_id: str,
    vessel_update: VesselUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update vessel data"""
    result = await db.execute(
        select(Vessel).where(Vessel.id == vessel_id)
    )
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")

    # Update fields
    for field, value in vessel_update.model_dump(exclude_unset=True).items():
        setattr(vessel, field, value)

    await db.commit()
    await db.refresh(vessel)
    return vessel

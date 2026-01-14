from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime

from src.database.connection import get_db
from src.database.models import Prediction
from src.schemas.prediction import (
    PredictionDetail,
    AnalyzeRequest,
    ConfirmOutcomeRequest
)
from src.services.prediction_engine import PredictionEngine
from src.services.learning_service import LearningService

router = APIRouter()


@router.post("/analyze")
async def analyze_vessel(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger prediction analysis for a vessel.
    Runs full pipeline: traditional + RAG + AI.
    """
    engine = PredictionEngine(db)
    predictions = await engine.analyze_vessel_with_ai(request.vessel_id)
    return {
        "vessel_id": request.vessel_id,
        "predictions_created": len(predictions),
        "predictions": predictions
    }


@router.get("/active", response_model=List[PredictionDetail])
async def list_active_predictions(
    db: AsyncSession = Depends(get_db)
):
    """Get all active predictions"""
    result = await db.execute(
        select(Prediction).where(Prediction.status == 'active')
    )
    predictions = result.scalars().all()

    return [
        PredictionDetail(
            id=p.id,
            vessel_id=p.vessel_id,
            vessel_name=p.vessel.name,
            terminal_id=p.terminal_id,
            terminal_name=p.terminal.name,
            confidence_score=p.confidence_score,
            distance_to_terminal_km=p.distance_to_terminal_km,
            eta_hours=p.eta_hours,
            predicted_arrival=p.predicted_arrival,
            proximity_score=p.proximity_score,
            speed_score=p.speed_score,
            heading_score=p.heading_score,
            historical_similarity_score=p.historical_similarity_score,
            ai_confidence_adjustment=p.ai_confidence_adjustment,
            ai_reasoning=p.ai_reasoning,
            status=p.status,
            prediction_time=p.prediction_time
        ) for p in predictions
    ]


@router.get("/{prediction_id}", response_model=PredictionDetail)
async def get_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific prediction details"""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    return PredictionDetail(
        id=prediction.id,
        vessel_id=prediction.vessel_id,
        vessel_name=prediction.vessel.name,
        terminal_id=prediction.terminal_id,
        terminal_name=prediction.terminal.name,
        confidence_score=prediction.confidence_score,
        distance_to_terminal_km=prediction.distance_to_terminal_km,
        eta_hours=prediction.eta_hours,
        predicted_arrival=prediction.predicted_arrival,
        proximity_score=prediction.proximity_score,
        speed_score=prediction.speed_score,
        heading_score=prediction.heading_score,
        historical_similarity_score=prediction.historical_similarity_score,
        ai_confidence_adjustment=prediction.ai_confidence_adjustment,
        ai_reasoning=prediction.ai_reasoning,
        status=prediction.status,
        prediction_time=prediction.prediction_time
    )


@router.post("/{prediction_id}/confirm")
async def confirm_prediction_outcome(
    prediction_id: int,
    request: ConfirmOutcomeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Confirm prediction outcome (triggers learning)"""
    learning = LearningService(db)
    await learning.process_prediction_outcome(
        prediction_id=prediction_id,
        actual_arrival_time=request.actual_arrival_time or datetime.utcnow(),
        outcome_status=request.status
    )

    return {"message": f"Prediction {prediction_id} outcome processed"}

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.database.connection import get_db
from src.database.models import Prediction, Vessel, Terminal, VectorEmbedding

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check with database connectivity"""
    try:
        await db.execute(select(func.count()).select_from(Vessel))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """System metrics"""
    # Count vessels
    vessel_count = await db.scalar(select(func.count()).select_from(Vessel))

    # Count terminals
    terminal_count = await db.scalar(select(func.count()).select_from(Terminal))

    # Count predictions by status
    active_predictions = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.status == 'active')
    )
    confirmed_predictions = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.status == 'confirmed')
    )

    # Count embeddings
    embedding_count = await db.scalar(select(func.count()).select_from(VectorEmbedding))

    return {
        "vessels": vessel_count,
        "terminals": terminal_count,
        "predictions": {
            "active": active_predictions,
            "confirmed": confirmed_predictions
        },
        "embeddings": embedding_count
    }


@router.get("/accuracy")
async def get_accuracy_stats(db: AsyncSession = Depends(get_db)):
    """Prediction accuracy statistics"""
    # Get confirmed predictions with accuracy scores
    result = await db.execute(
        select(Prediction).where(
            Prediction.status == 'confirmed',
            Prediction.accuracy_score.isnot(None)
        )
    )
    predictions = result.scalars().all()

    if not predictions:
        return {"message": "No confirmed predictions yet"}

    accuracies = [p.accuracy_score for p in predictions]
    avg_accuracy = sum(accuracies) / len(accuracies)

    return {
        "total_confirmed": len(predictions),
        "average_accuracy": avg_accuracy,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies)
    }

"""
Background service to automatically detect when vessels arrive at terminals.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
import logging

from src.database.models import Prediction, Vessel
from src.services.learning_service import LearningService
from src.utils.geo import haversine_distance

logger = logging.getLogger(__name__)


class OutcomeDetector:
    """
    Detects when active predictions should be confirmed or expired.
    Runs periodically to check vessel positions.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.learning_service = LearningService(db_session)

    async def check_active_predictions(self):
        """
        Check all active predictions for outcomes.

        For each active prediction:
        1. Check if vessel is at predicted terminal (distance < threshold)
        2. If yes, confirm prediction
        3. If ETA passed by >12 hours, expire prediction
        """
        # Get all active predictions
        result = await self.db.execute(
            select(Prediction).where(Prediction.status == 'active')
        )
        predictions = result.scalars().all()

        logger.info(f"Checking {len(predictions)} active predictions")

        for prediction in predictions:
            await self._check_prediction(prediction)

    async def _check_prediction(self, prediction: Prediction):
        """Check a single prediction for outcome"""
        # Get current vessel position
        result = await self.db.execute(
            select(Vessel).where(Vessel.id == prediction.vessel_id)
        )
        vessel = result.scalar_one_or_none()

        if not vessel:
            logger.warning(f"Vessel {prediction.vessel_id} not found")
            return

        # Calculate distance to terminal
        distance = haversine_distance(
            vessel.current_lat,
            vessel.current_lon,
            prediction.terminal.lat,
            prediction.terminal.lon
        )

        # Check if arrived (within terminal approach zone)
        arrival_threshold_km = prediction.terminal.approach_zone_radius_km * 0.3

        if distance <= arrival_threshold_km and vessel.speed < 5.0:
            # Vessel is at terminal and slow/stopped
            logger.info(
                f"Vessel {vessel.name} arrived at {prediction.terminal.name} "
                f"(prediction {prediction.id} confirmed)"
            )
            await self.learning_service.process_prediction_outcome(
                prediction_id=prediction.id,
                actual_arrival_time=datetime.utcnow(),
                outcome_status='confirmed'
            )

        # Check if prediction expired
        elif prediction.predicted_arrival:
            time_since_eta = datetime.utcnow() - prediction.predicted_arrival

            if time_since_eta > timedelta(hours=12):
                # ETA passed by more than 12 hours, prediction likely incorrect
                logger.info(
                    f"Prediction {prediction.id} expired "
                    f"(ETA passed by {time_since_eta.total_seconds() / 3600:.1f}h)"
                )
                await self.learning_service.process_prediction_outcome(
                    prediction_id=prediction.id,
                    actual_arrival_time=prediction.predicted_arrival,
                    outcome_status='expired'
                )

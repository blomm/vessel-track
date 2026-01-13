from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime
import logging

from src.database.models import (
    Prediction,
    VectorEmbedding,
    TerminalApproachBehavior,
    BehaviorEvent,
    Terminal
)
from src.services.embedding_service import EmbeddingService
from src.config import settings

logger = logging.getLogger(__name__)


class LearningService:
    """
    Handles continuous learning from prediction outcomes.
    Updates RAG system and approach behaviors based on actual results.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.embedding_service = EmbeddingService()

    async def process_prediction_outcome(
        self,
        prediction_id: int,
        actual_arrival_time: datetime,
        outcome_status: str = 'confirmed'
    ):
        """
        Called when vessel arrives at terminal (or prediction expires).

        Steps:
        1. Update prediction with actual outcome
        2. Calculate accuracy metrics
        3. Generate embedding of prediction + outcome
        4. Store in vector_embeddings for RAG
        5. Update terminal approach behaviors
        6. Log for metrics

        Args:
            prediction_id: ID of prediction to process
            actual_arrival_time: When vessel actually arrived
            outcome_status: 'confirmed', 'incorrect', or 'expired'
        """
        # Get prediction
        result = await self.db.execute(
            select(Prediction).where(Prediction.id == prediction_id)
        )
        prediction = result.scalar_one_or_none()

        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")

        if prediction.status != 'active':
            logger.warning(f"Prediction {prediction_id} already processed")
            return

        # Update prediction
        prediction.actual_arrival_time = actual_arrival_time
        prediction.status = outcome_status

        # Calculate accuracy
        if outcome_status == 'confirmed':
            accuracy = self._calculate_accuracy(prediction, actual_arrival_time)
            prediction.accuracy_score = accuracy

            logger.info(
                f"Prediction {prediction_id} confirmed with {accuracy:.1%} accuracy"
            )
        else:
            prediction.accuracy_score = 0.0
            logger.info(f"Prediction {prediction_id} marked as {outcome_status}")

        # Generate and store embedding
        await self._store_prediction_embedding(prediction)

        # Update approach behaviors (only for successful predictions)
        if outcome_status == 'confirmed':
            await self._update_approach_behaviors(prediction)

        await self.db.commit()

    def _calculate_accuracy(
        self,
        prediction: Prediction,
        actual_arrival: datetime
    ) -> float:
        """
        Calculate accuracy score (0.0-1.0).

        Factors:
        - Time accuracy: How close was the ETA?
        - Terminal accuracy: Was it the correct terminal? (binary)
        """
        if not prediction.predicted_arrival:
            return 0.5  # No ETA prediction

        # Time difference in hours
        time_diff = abs(
            (actual_arrival - prediction.predicted_arrival).total_seconds() / 3600
        )

        # Time accuracy: Exponential decay
        # 0 hours off = 1.0
        # 2 hours off = 0.85
        # 4 hours off = 0.64
        # 12 hours off = 0.20
        time_accuracy = 1.0 / (1.0 + (time_diff / 4.0) ** 2)

        # Terminal is correct (we're in this function), so terminal_accuracy = 1.0

        # Combined accuracy
        accuracy = time_accuracy

        return max(0.0, min(1.0, accuracy))

    async def _store_prediction_embedding(self, prediction: Prediction):
        """
        Generate and store embedding of prediction outcome for RAG.
        """
        # Generate embedding
        embedding_vector = await self.embedding_service.embed_prediction_outcome(
            prediction
        )

        # Store in database
        vector_embedding = VectorEmbedding(
            content_type='prediction',
            content_id=prediction.id,
            text_content=self.embedding_service._prediction_to_text(prediction),
            embedding=embedding_vector,
            metadata={
                'vessel_id': prediction.vessel_id,
                'terminal_id': prediction.terminal_id,
                'confidence': prediction.confidence_score,
                'accuracy': prediction.accuracy_score,
                'status': prediction.status
            }
        )

        self.db.add(vector_embedding)
        logger.debug(f"Stored embedding for prediction {prediction.id}")

    async def _update_approach_behaviors(self, prediction: Prediction):
        """
        Update learned approach behavior patterns based on successful prediction.

        Uses exponential moving average to incorporate new observations.
        """
        # Find existing behavior for this vessel+terminal
        result = await self.db.execute(
            select(TerminalApproachBehavior).where(
                and_(
                    TerminalApproachBehavior.terminal_id == prediction.terminal_id,
                    TerminalApproachBehavior.vessel_id == prediction.vessel_id
                )
            )
        )
        behavior = result.scalar_one_or_none()

        if behavior:
            # Update existing pattern (exponential moving average)
            alpha = 0.3  # Weight for new observation
            behavior.observation_count += 1

            # Update distance
            behavior.approach_distance_km = (
                (1 - alpha) * behavior.approach_distance_km +
                alpha * prediction.distance_to_terminal_km
            )

            # Update speed ranges
            if prediction.vessel_speed:
                if behavior.typical_speed_range_min:
                    behavior.typical_speed_range_min = min(
                        behavior.typical_speed_range_min,
                        prediction.vessel_speed
                    )
                else:
                    behavior.typical_speed_range_min = prediction.vessel_speed

                if behavior.typical_speed_range_max:
                    behavior.typical_speed_range_max = max(
                        behavior.typical_speed_range_max,
                        prediction.vessel_speed
                    )
                else:
                    behavior.typical_speed_range_max = prediction.vessel_speed

            # Update confidence (based on accuracy)
            if prediction.accuracy_score:
                if behavior.confidence:
                    behavior.confidence = (
                        (1 - alpha) * behavior.confidence +
                        alpha * prediction.accuracy_score
                    )
                else:
                    behavior.confidence = prediction.accuracy_score

            behavior.last_observed = datetime.utcnow()
            behavior.updated_at = datetime.utcnow()

        else:
            # Create new behavior pattern
            behavior = TerminalApproachBehavior(
                terminal_id=prediction.terminal_id,
                vessel_id=prediction.vessel_id,
                approach_distance_km=prediction.distance_to_terminal_km,
                typical_speed_range_min=prediction.vessel_speed,
                typical_speed_range_max=prediction.vessel_speed,
                typical_heading_range_min=prediction.vessel_heading,
                typical_heading_range_max=prediction.vessel_heading,
                observation_count=1,
                confidence=prediction.accuracy_score,
                last_observed=datetime.utcnow()
            )
            self.db.add(behavior)

        logger.debug(
            f"Updated approach behavior for vessel {prediction.vessel_id} → "
            f"terminal {prediction.terminal_id}"
        )

    async def detect_behavior_events(
        self,
        vessel_id: str,
        current_state: dict,
        previous_state: dict
    ):
        """
        Detect significant behavior changes (speed/course changes).

        Called whenever vessel state updates.

        Args:
            vessel_id: Vessel ID
            current_state: Current vessel state (lat, lon, speed, heading)
            previous_state: Previous vessel state
        """
        # Speed change detection
        speed_diff = abs(
            current_state.get('speed', 0) - previous_state.get('speed', 0)
        )

        if speed_diff >= 3.0:  # Threshold: 3 knots
            await self._create_behavior_event(
                vessel_id=vessel_id,
                event_type='speed_change',
                lat=current_state['lat'],
                lon=current_state['lon'],
                speed_before=previous_state.get('speed'),
                speed_after=current_state['speed'],
                heading_before=previous_state.get('heading'),
                heading_after=current_state['heading'],
                magnitude=speed_diff
            )

        # Course change detection
        if previous_state.get('heading') and current_state.get('heading'):
            from src.utils.geo import angular_difference
            heading_diff = angular_difference(
                previous_state['heading'],
                current_state['heading']
            )

            if heading_diff >= 30.0:  # Threshold: 30 degrees
                await self._create_behavior_event(
                    vessel_id=vessel_id,
                    event_type='course_change',
                    lat=current_state['lat'],
                    lon=current_state['lon'],
                    speed_before=previous_state.get('speed'),
                    speed_after=current_state['speed'],
                    heading_before=previous_state.get('heading'),
                    heading_after=current_state['heading'],
                    magnitude=heading_diff
                )

        # Stop detection
        if previous_state.get('speed', 0) > 3.0 and current_state.get('speed', 0) < 1.0:
            await self._create_behavior_event(
                vessel_id=vessel_id,
                event_type='stop',
                lat=current_state['lat'],
                lon=current_state['lon'],
                speed_before=previous_state.get('speed'),
                speed_after=current_state['speed'],
                heading_before=previous_state.get('heading'),
                heading_after=current_state['heading'],
                magnitude=previous_state.get('speed', 0)
            )

    async def _create_behavior_event(
        self,
        vessel_id: str,
        event_type: str,
        lat: float,
        lon: float,
        speed_before: Optional[float],
        speed_after: Optional[float],
        heading_before: Optional[float],
        heading_after: Optional[float],
        magnitude: float
    ):
        """Create a behavior event record"""
        from src.utils.geo import haversine_distance

        # Find nearest terminal
        result = await self.db.execute(select(Terminal))
        terminals = result.scalars().all()

        nearest_terminal = None
        min_distance = float('inf')

        for terminal in terminals:
            distance = haversine_distance(lat, lon, terminal.lat, terminal.lon)
            if distance < min_distance:
                min_distance = distance
                nearest_terminal = terminal

        # Create event
        event = BehaviorEvent(
            vessel_id=vessel_id,
            event_type=event_type,
            lat=lat,
            lon=lon,
            speed_before=speed_before,
            speed_after=speed_after,
            heading_before=heading_before,
            heading_after=heading_after,
            magnitude=magnitude,
            nearest_terminal_id=nearest_terminal.id if nearest_terminal else None,
            distance_to_terminal_km=min_distance if nearest_terminal else None
        )

        self.db.add(event)
        await self.db.commit()

        logger.info(
            f"Behavior event detected: {event_type} for vessel {vessel_id} "
            f"(magnitude: {magnitude:.1f})"
        )

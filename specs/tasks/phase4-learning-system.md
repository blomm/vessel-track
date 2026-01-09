# Phase 4: Learning System

**Duration**: Days 18-21
**Goal**: Implement feedback loop for continuous learning from prediction outcomes

---

## 4.1. Learning Service

### Create `service/src/services/learning_service.py`:

```python
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
    Vessel
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
```

---

## 4.2. Automated Outcome Detection

### Create `service/src/services/outcome_detector.py`:

```python
"""
Background service to automatically detect when vessels arrive at terminals.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime, timedelta
import logging

from src.database.models import Prediction, Vessel, Terminal
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
```

---

## 4.3. Background Task Runner

### Create `service/src/services/background_tasks.py`:

```python
"""
Background tasks for periodic operations.
"""
import asyncio
import logging
from datetime import datetime

from src.database.connection import AsyncSessionLocal
from src.services.outcome_detector import OutcomeDetector
from src.config import settings

logger = logging.getLogger(__name__)


async def run_outcome_detection():
    """
    Periodically check for prediction outcomes.
    Runs every 5 minutes.
    """
    logger.info("Starting outcome detection background task")

    while True:
        try:
            async with AsyncSessionLocal() as session:
                detector = OutcomeDetector(session)
                await detector.check_active_predictions()
                logger.info("Outcome detection cycle completed")

        except Exception as e:
            logger.error(f"Error in outcome detection: {e}")

        # Wait 5 minutes
        await asyncio.sleep(300)


async def run_all_background_tasks():
    """Run all background tasks concurrently"""
    await asyncio.gather(
        run_outcome_detection(),
        # Add more background tasks here
    )
```

---

## Verification Checklist

- [ ] `LearningService` implemented
- [ ] Prediction outcome processing works
- [ ] Accuracy calculation formula implemented
- [ ] Embeddings generated for outcomes
- [ ] Approach behaviors update correctly
- [ ] Behavior event detection implemented
- [ ] Speed change events detected (threshold: 3 knots)
- [ ] Course change events detected (threshold: 30°)
- [ ] Stop events detected
- [ ] `OutcomeDetector` implemented
- [ ] Automatic arrival detection works
- [ ] Prediction expiration logic works
- [ ] Background tasks runner implemented
- [ ] Can manually trigger learning for test prediction

---

## Testing

### Manual Test:

```bash
cd service

# Test learning service
poetry run python -c "
import asyncio
from datetime import datetime, timedelta
from src.database.connection import AsyncSessionLocal
from src.services.learning_service import LearningService

async def test():
    async with AsyncSessionLocal() as session:
        learning = LearningService(session)

        # Simulate prediction outcome
        prediction_id = 1  # Use actual prediction ID
        actual_arrival = datetime.utcnow()

        await learning.process_prediction_outcome(
            prediction_id=prediction_id,
            actual_arrival_time=actual_arrival,
            outcome_status='confirmed'
        )

        print('✓ Prediction outcome processed')
        print('✓ Embedding generated and stored')
        print('✓ Approach behaviors updated')

asyncio.run(test())
"

# Test outcome detector
poetry run python -c "
import asyncio
from src.database.connection import AsyncSessionLocal
from src.services.outcome_detector import OutcomeDetector

async def test():
    async with AsyncSessionLocal() as session:
        detector = OutcomeDetector(session)
        await detector.check_active_predictions()
        print('✓ Checked all active predictions')

asyncio.run(test())
"
```

---

## Integration with FastAPI

Add background tasks to main application:

### Update `service/src/main.py`:

```python
from fastapi import FastAPI
import asyncio

from src.services.background_tasks import run_all_background_tasks

app = FastAPI(title="Vessel Track API")

# Background task handle
background_task = None

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global background_task
    background_task = asyncio.create_task(run_all_background_tasks())
    logger.info("Background tasks started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks"""
    global background_task
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            logger.info("Background tasks stopped")
```

---

## Next Steps

Once this phase is complete, move to **Phase 5: API & WebSocket** where we'll implement:
- REST API endpoints for vessels, terminals, predictions
- WebSocket manager for real-time updates
- Admin endpoints for metrics and learning triggers
- Complete API documentation

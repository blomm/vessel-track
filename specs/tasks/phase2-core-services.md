# Phase 2: Core Services

**Duration**: Days 5-12
**Goal**: Implement geographic utilities, traditional prediction engine, embedding service, and RAG service

---

## 2.1. Geographic Utilities

### Create `service/src/utils/geo.py`:

```python
from math import radians, cos, sin, asin, sqrt, atan2, degrees
from typing import Tuple

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth.
    Uses Haversine formula.

    Args:
        lat1, lon1: First point coordinates (decimal degrees)
        lat2, lon2: Second point coordinates (decimal degrees)

    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Earth's radius in kilometers
    km = 6371 * c
    return km


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Start point coordinates (decimal degrees)
        lat2, lon2: End point coordinates (decimal degrees)

    Returns:
        Bearing in degrees (0-360), where 0° is North
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    initial_bearing = atan2(x, y)
    bearing = (degrees(initial_bearing) + 360) % 360

    return bearing


def angular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest angle between two headings.

    Args:
        angle1, angle2: Angles in degrees (0-360)

    Returns:
        Smallest angular difference in degrees (0-180)
    """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def nautical_miles_to_km(nm: float) -> float:
    """Convert nautical miles to kilometers"""
    return nm * 1.852


def km_to_nautical_miles(km: float) -> float:
    """Convert kilometers to nautical miles"""
    return km / 1.852
```

### Create tests `service/tests/test_utils/test_geo.py`:

```python
import pytest
from src.utils.geo import haversine_distance, calculate_bearing, angular_difference

def test_haversine_distance():
    # New York to London: ~5570 km
    ny_lat, ny_lon = 40.7128, -74.0060
    london_lat, london_lon = 51.5074, -0.1278

    distance = haversine_distance(ny_lat, ny_lon, london_lat, london_lon)
    assert 5500 < distance < 5600  # Approximate

def test_calculate_bearing():
    # North bearing from equator
    bearing = calculate_bearing(0, 0, 1, 0)
    assert 0 <= bearing < 10  # Approximately north

def test_angular_difference():
    assert angular_difference(10, 350) == 20  # Wraps around 360
    assert angular_difference(90, 270) == 180
    assert angular_difference(45, 45) == 0
```

---

## 2.2. Traditional Prediction Engine

### Create `service/src/services/prediction_engine.py`:

```python
from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging

from src.database.models import Vessel, Terminal, Prediction
from src.utils.geo import haversine_distance, calculate_bearing, angular_difference
from src.config import settings

logger = logging.getLogger(__name__)

class PredictionResult:
    """Container for prediction results"""
    def __init__(
        self,
        terminal_id: int,
        terminal_name: str,
        confidence_score: float,
        proximity_score: float,
        speed_score: float,
        heading_score: float,
        distance_km: float,
        eta_hours: Optional[float],
    ):
        self.terminal_id = terminal_id
        self.terminal_name = terminal_name
        self.confidence_score = confidence_score
        self.proximity_score = proximity_score
        self.speed_score = speed_score
        self.heading_score = heading_score
        self.distance_km = distance_km
        self.eta_hours = eta_hours


class PredictionEngine:
    """
    Traditional algorithmic prediction engine.
    Combines proximity, speed, and heading analysis.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def analyze_vessel(self, vessel_id: str) -> List[PredictionResult]:
        """
        Main entry point for prediction analysis.

        Steps:
        1. Get vessel current state
        2. Find terminals within max distance
        3. For each terminal, calculate scores
        4. Return predictions above minimum confidence threshold
        """
        # Get vessel
        result = await self.db.execute(
            select(Vessel).where(Vessel.id == vessel_id)
        )
        vessel = result.scalar_one_or_none()
        if not vessel:
            raise ValueError(f"Vessel {vessel_id} not found")

        # Get all active terminals
        result = await self.db.execute(select(Terminal))
        terminals = result.scalars().all()

        # Calculate predictions for each terminal
        predictions = []
        for terminal in terminals:
            distance_km = haversine_distance(
                vessel.current_lat,
                vessel.current_lon,
                terminal.lat,
                terminal.lon
            )

            # Skip if too far
            if distance_km > settings.PREDICTION_MAX_DISTANCE_KM:
                continue

            # Calculate scores
            proximity_score = self.calculate_proximity_score(distance_km)
            speed_score = self.calculate_speed_score(vessel.speed)
            heading_score = self.calculate_heading_score(
                vessel.current_lat,
                vessel.current_lon,
                vessel.heading,
                terminal.lat,
                terminal.lon
            )

            # Base confidence (0.0-1.0)
            base_confidence = (
                proximity_score * 0.4 +
                speed_score * 0.3 +
                heading_score * 0.3
            )

            # Skip low confidence
            if base_confidence < settings.PREDICTION_MIN_CONFIDENCE:
                continue

            # Calculate ETA
            eta_hours = None
            if vessel.speed and vessel.speed > 1.0:
                eta_hours = distance_km / (vessel.speed * 1.852)  # knots to km/h

            prediction = PredictionResult(
                terminal_id=terminal.id,
                terminal_name=terminal.name,
                confidence_score=base_confidence,
                proximity_score=proximity_score,
                speed_score=speed_score,
                heading_score=heading_score,
                distance_km=distance_km,
                eta_hours=eta_hours,
            )
            predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence_score, reverse=True)

        logger.info(
            f"Generated {len(predictions)} predictions for vessel {vessel_id}"
        )

        return predictions

    def calculate_proximity_score(self, distance_km: float) -> float:
        """
        Calculate proximity score based on distance.

        Uses exponential decay:
        - At 0 km: 1.0
        - At 50 km: ~0.5
        - At 100 km: ~0.2
        - At 500 km: ~0.04

        Formula: 1.0 / (1.0 + (distance_km / 50) ** 2)
        """
        score = 1.0 / (1.0 + (distance_km / 50.0) ** 2)
        return max(0.0, min(1.0, score))

    def calculate_speed_score(self, speed_knots: Optional[float]) -> float:
        """
        Calculate speed score based on vessel speed.

        LNG tankers typically:
        - Transit speed: 15-18 knots
        - Approach speed: 8-15 knots (OPTIMAL for terminal approach)
        - Berthing speed: 3-8 knots
        - Too slow: < 3 knots (drifting, not approaching)
        - Too fast: > 18 knots (passing by, not approaching)

        Returns:
            Score from 0.0 to 1.0
        """
        if speed_knots is None:
            return 0.5  # Unknown speed = neutral

        if speed_knots < 1.0:
            return 0.1  # Drifting or stationary

        if 8.0 <= speed_knots <= 15.0:
            return 1.0  # Optimal approach speed

        if 3.0 <= speed_knots < 8.0:
            return 0.7  # Slow approach / berthing

        if 15.0 < speed_knots <= 18.0:
            return 0.6  # Transit speed, might be approaching

        if speed_knots > 18.0:
            return 0.2  # Too fast, likely passing by

        return 0.3  # Default for other cases

    def calculate_heading_score(
        self,
        vessel_lat: float,
        vessel_lon: float,
        vessel_heading: Optional[float],
        terminal_lat: float,
        terminal_lon: float,
    ) -> float:
        """
        Calculate heading score based on vessel heading vs bearing to terminal.

        Returns:
            Score from 0.0 to 1.0
        """
        if vessel_heading is None:
            return 0.5  # Unknown heading = neutral

        # Calculate bearing from vessel to terminal
        bearing_to_terminal = calculate_bearing(
            vessel_lat, vessel_lon, terminal_lat, terminal_lon
        )

        # Calculate angular difference
        angle_diff = angular_difference(vessel_heading, bearing_to_terminal)

        # Score based on alignment
        if angle_diff <= 15:
            return 1.0  # Very aligned
        elif angle_diff <= 30:
            return 0.8  # Moderately aligned
        elif angle_diff <= 45:
            return 0.6  # Somewhat aligned
        elif angle_diff <= 90:
            return 0.3  # Perpendicular
        else:
            return 0.1  # Heading away
```

---

## 2.3. Embedding Service

### Create `service/src/services/embedding_service.py`:

```python
import openai
from typing import List
import logging

from src.config import settings
from src.database.models import VesselJourney, Prediction

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


class EmbeddingService:
    """
    Generate vector embeddings using OpenAI text-embedding-3-small.
    Used for RAG similarity search.
    """

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for arbitrary text.

        Args:
            text: Text to embed

        Returns:
            List of 1536 floats (embedding vector)
        """
        try:
            response = await client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_journey(self, journey: VesselJourney) -> List[float]:
        """
        Convert journey to text and generate embedding.

        Args:
            journey: VesselJourney database model

        Returns:
            Embedding vector
        """
        text = self._journey_to_text(journey)
        return await self.embed_text(text)

    async def embed_prediction_outcome(self, prediction: Prediction) -> List[float]:
        """
        Embed prediction with its outcome for learning.

        Args:
            prediction: Prediction database model (with outcome)

        Returns:
            Embedding vector
        """
        text = self._prediction_to_text(prediction)
        return await self.embed_text(text)

    def _journey_to_text(self, journey: VesselJourney) -> str:
        """
        Convert journey to structured text for embedding.

        Example:
        "LNG tanker Pacific Energy traveled from Ras Laffan LNG (Qatar)
        to Tokyo Gas Negishi (Japan). Journey took 168 hours covering
        5400 nautical miles at average speed 14.5 knots. Vessel approached
        destination from southwest at typical LNG tanker approach speed."
        """
        origin_name = journey.origin.name if journey.origin else "Unknown"
        destination_name = journey.destination.name

        text = (
            f"LNG tanker {journey.vessel.name} traveled from {origin_name} "
            f"to {destination_name}. "
        )

        if journey.duration_hours:
            text += f"Journey took {journey.duration_hours:.1f} hours "

        if journey.distance_nm:
            text += f"covering {journey.distance_nm:.0f} nautical miles "

        if journey.avg_speed:
            text += f"at average speed {journey.avg_speed:.1f} knots. "

        text += (
            f"Destination terminal: {journey.destination.name} "
            f"({journey.destination.country}), "
            f"terminal type: {journey.destination.terminal_type}."
        )

        return text

    def _prediction_to_text(self, prediction: Prediction) -> str:
        """
        Convert prediction + outcome to text for learning.

        Example:
        "Prediction made 36 hours before arrival: Arctic Spirit predicted
        to arrive at Sabine Pass LNG with 87% confidence. Factors: proximity
        score 0.82, speed 11 knots, heading aligned. Outcome: Confirmed arrival
        within 2 hours of ETA. Accuracy: 95%."
        """
        hours_before = None
        if prediction.predicted_arrival and prediction.actual_arrival_time:
            delta = prediction.actual_arrival_time - prediction.prediction_time
            hours_before = delta.total_seconds() / 3600

        text = f"Prediction for {prediction.vessel.name} to {prediction.terminal.name}: "

        if hours_before:
            text += f"Made {hours_before:.1f} hours before arrival. "

        text += f"Confidence: {prediction.confidence_score:.2f}. "
        text += f"Distance: {prediction.distance_to_terminal_km:.1f}km. "
        text += f"Speed: {prediction.vessel_speed:.1f} knots. "

        text += (
            f"Scores - Proximity: {prediction.proximity_score:.2f}, "
            f"Speed: {prediction.speed_score:.2f}, "
            f"Heading: {prediction.heading_score:.2f}. "
        )

        if prediction.ai_reasoning:
            text += f"AI Analysis: {prediction.ai_reasoning} "

        text += f"Outcome: {prediction.status}. "

        if prediction.accuracy_score:
            text += f"Accuracy: {prediction.accuracy_score:.2f}."

        return text
```

---

## 2.4. RAG Service

### Create `service/src/services/rag_service.py`:

```python
from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, text
import logging

from src.database.models import (
    VesselJourney,
    VectorEmbedding,
    Vessel,
    Terminal,
    TerminalApproachBehavior
)
from src.services.embedding_service import EmbeddingService
from src.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Uses pgvector for similarity search.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.embedding_service = EmbeddingService()

    async def find_similar_journeys(
        self,
        vessel: Vessel,
        terminal: Terminal,
        limit: int = 5
    ) -> List[Tuple[VesselJourney, float]]:
        """
        Find historical journeys similar to current situation.

        Args:
            vessel: Current vessel
            terminal: Target terminal
            limit: Max number of results

        Returns:
            List of (journey, similarity_score) tuples
        """
        # Create query text for current situation
        query_text = (
            f"LNG tanker {vessel.name} approaching {terminal.name} "
            f"({terminal.country}). Current speed {vessel.speed} knots. "
            f"Terminal type: {terminal.terminal_type}."
        )

        # Generate embedding
        query_embedding = await self.embedding_service.embed_text(query_text)

        # Convert to PostgreSQL array format
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # Vector similarity query using pgvector
        query = text(f"""
            SELECT
                vj.id,
                vj.vessel_id,
                vj.origin_terminal_id,
                vj.destination_terminal_id,
                vj.departure_time,
                vj.arrival_time,
                vj.duration_hours,
                vj.avg_speed,
                vj.distance_nm,
                ve.embedding <=> :embedding::vector AS similarity
            FROM vessel_journeys vj
            JOIN vector_embeddings ve
                ON ve.content_type = 'journey'
                AND ve.content_id = vj.id
            WHERE vj.destination_terminal_id = :terminal_id
                AND vj.completed = true
            ORDER BY similarity ASC
            LIMIT :limit
        """)

        result = await self.db.execute(
            query,
            {
                "embedding": embedding_str,
                "terminal_id": terminal.id,
                "limit": limit
            }
        )

        # Process results
        similar_journeys = []
        for row in result:
            # Fetch full journey object
            journey_result = await self.db.execute(
                select(VesselJourney).where(VesselJourney.id == row.id)
            )
            journey = journey_result.scalar_one_or_none()

            if journey:
                similarity_score = float(row.similarity)
                similar_journeys.append((journey, similarity_score))

        logger.info(
            f"Found {len(similar_journeys)} similar journeys for "
            f"{vessel.name} → {terminal.name}"
        )

        return similar_journeys

    async def get_terminal_approach_patterns(
        self,
        terminal_id: int,
        vessel_id: Optional[str] = None
    ) -> List[TerminalApproachBehavior]:
        """
        Get learned approach behaviors for a terminal.

        Args:
            terminal_id: Terminal to get patterns for
            vessel_id: Optional - get patterns specific to this vessel

        Returns:
            List of approach behavior patterns
        """
        query = select(TerminalApproachBehavior).where(
            TerminalApproachBehavior.terminal_id == terminal_id
        )

        if vessel_id:
            # Vessel-specific patterns first
            query = query.where(TerminalApproachBehavior.vessel_id == vessel_id)
        else:
            # Generic patterns (vessel_id is NULL)
            query = query.where(TerminalApproachBehavior.vessel_id.is_(None))

        query = query.order_by(TerminalApproachBehavior.confidence.desc())

        result = await self.db.execute(query)
        behaviors = result.scalars().all()

        logger.debug(
            f"Retrieved {len(behaviors)} approach patterns for terminal {terminal_id}"
        )

        return list(behaviors)

    def calculate_historical_similarity_score(
        self,
        similar_journeys: List[Tuple[VesselJourney, float]]
    ) -> float:
        """
        Calculate a score based on RAG retrieval results.

        Args:
            similar_journeys: List of (journey, similarity_score) tuples

        Returns:
            Score from 0.0 to 0.3 (to add to base prediction score)
        """
        if not similar_journeys:
            return 0.0

        # Get average similarity (pgvector cosine distance: lower is more similar)
        avg_similarity = sum(sim for _, sim in similar_journeys) / len(similar_journeys)

        # Convert similarity to score
        # Cosine distance 0.0 = identical, 2.0 = opposite
        # We want: distance 0.0-0.3 = high score, >1.0 = low score
        if avg_similarity < settings.RAG_SIMILARITY_THRESHOLD:
            # Very similar historical patterns found
            score = 0.3 * (1.0 - (avg_similarity / settings.RAG_SIMILARITY_THRESHOLD))
        else:
            # Not very similar
            score = 0.0

        return max(0.0, min(0.3, score))
```

---

## Verification Checklist

- [ ] Geographic utilities implemented (`geo.py`)
- [ ] Distance calculations tested (Haversine formula)
- [ ] Bearing calculations tested
- [ ] `PredictionEngine` class implemented
- [ ] Proximity scoring formula implemented (exponential decay)
- [ ] Speed scoring logic implemented (optimal 8-15 knots)
- [ ] Heading scoring logic implemented (angular difference)
- [ ] `EmbeddingService` implemented with OpenAI client
- [ ] Journey-to-text conversion implemented
- [ ] Prediction-to-text conversion implemented
- [ ] `RAGService` implemented with pgvector queries
- [ ] Vector similarity search tested
- [ ] Can generate embeddings for test data
- [ ] Can retrieve similar journeys from database
- [ ] Unit tests pass for all services

---

## Testing

### Manual Testing:

```bash
cd service

# Test geographic utilities
poetry run pytest tests/test_utils/test_geo.py -v

# Test prediction engine with mock data
poetry run python -c "
import asyncio
from src.database.connection import AsyncSessionLocal
from src.services.prediction_engine import PredictionEngine

async def test():
    async with AsyncSessionLocal() as session:
        engine = PredictionEngine(session)
        predictions = await engine.analyze_vessel('lng-001')
        for p in predictions:
            print(f'{p.terminal_name}: {p.confidence_score:.2f}')

asyncio.run(test())
"
```

---

## Next Steps

Once this phase is complete, move to **Phase 3: AI Integration** where we'll implement:
- GPT-4o analysis service
- Prompt engineering for confidence adjustment
- Integration of AI with traditional prediction engine
- Complete end-to-end prediction pipeline

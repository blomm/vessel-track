from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging
from datetime import datetime, timedelta

from src.database.models import Vessel, Terminal, Prediction
from src.utils.geo import haversine_distance, calculate_bearing, angular_difference
from src.config import settings
from src.services.ai_service import AIService
from src.services.rag_service import RAGService
from src.services.slack_service import SlackService

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
        self.ai_service = AIService()
        self.rag_service = RAGService(db_session)

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

    async def analyze_vessel_with_ai(self, vessel_id: str) -> List[dict]:
        """
        Complete prediction pipeline with AI and RAG integration.

        Steps:
        1. Get vessel and nearby terminals
        2. Calculate traditional scores (proximity, speed, heading)
        3. Query RAG for similar historical journeys
        4. Get learned approach behaviors
        5. Send all context to GPT-4o for analysis
        6. Combine scores and create final predictions
        7. Store in database
        8. Trigger Slack notifications for high confidence

        Returns:
            List of prediction dictionaries sorted by confidence
        """
        # Get vessel
        result = await self.db.execute(
            select(Vessel).where(Vessel.id == vessel_id)
        )
        vessel = result.scalar_one_or_none()
        if not vessel:
            raise ValueError(f"Vessel {vessel_id} not found")

        # Get terminals
        result = await self.db.execute(select(Terminal))
        terminals = result.scalars().all()

        predictions = []
        slack_service = SlackService()

        for terminal in terminals:
            # Calculate distance
            distance_km = haversine_distance(
                vessel.current_lat, vessel.current_lon,
                terminal.lat, terminal.lon
            )

            # Skip if too far
            if distance_km > settings.PREDICTION_MAX_DISTANCE_KM:
                continue

            # Traditional scores
            proximity_score = self.calculate_proximity_score(distance_km)
            speed_score = self.calculate_speed_score(vessel.speed)
            heading_score = self.calculate_heading_score(
                vessel.current_lat, vessel.current_lon,
                vessel.heading, terminal.lat, terminal.lon
            )

            base_score = (
                proximity_score * 0.4 +
                speed_score * 0.3 +
                heading_score * 0.3
            )

            # RAG retrieval
            similar_journeys = await self.rag_service.find_similar_journeys(
                vessel, terminal, limit=5
            )
            historical_score = self.rag_service.calculate_historical_similarity_score(
                similar_journeys
            )

            # Get approach behaviors
            approach_behaviors = await self.rag_service.get_terminal_approach_patterns(
                terminal.id, vessel_id=vessel.id
            )

            # AI analysis
            ai_result = await self.ai_service.analyze_prediction(
                vessel=vessel,
                terminal=terminal,
                proximity_score=proximity_score,
                speed_score=speed_score,
                heading_score=heading_score,
                historical_matches=similar_journeys,
                approach_behaviors=approach_behaviors,
            )

            # Final confidence score
            final_score = base_score + historical_score + ai_result.confidence_adjustment
            final_score = max(0.0, min(1.0, final_score))

            # Skip low confidence
            if final_score < settings.PREDICTION_MIN_CONFIDENCE:
                continue

            # Calculate ETA
            eta_hours = None
            predicted_arrival = None
            if vessel.speed and vessel.speed > 1.0:
                eta_hours = distance_km / (vessel.speed * 1.852)
                predicted_arrival = datetime.utcnow() + timedelta(hours=eta_hours)

            # Create prediction record
            prediction = Prediction(
                vessel_id=vessel.id,
                terminal_id=terminal.id,
                vessel_lat=vessel.current_lat,
                vessel_lon=vessel.current_lon,
                vessel_speed=vessel.speed,
                vessel_heading=vessel.heading,
                confidence_score=final_score,
                distance_to_terminal_km=distance_km,
                eta_hours=eta_hours,
                predicted_arrival=predicted_arrival,
                proximity_score=proximity_score,
                speed_score=speed_score,
                heading_score=heading_score,
                historical_similarity_score=historical_score,
                ai_confidence_adjustment=ai_result.confidence_adjustment,
                ai_reasoning=ai_result.reasoning,
                status='active'
            )

            self.db.add(prediction)
            await self.db.flush()  # Get prediction ID

            predictions.append({
                'prediction_id': prediction.id,
                'terminal_name': terminal.name,
                'confidence': final_score,
                'eta_hours': eta_hours,
                'ai_reasoning': ai_result.reasoning,
                'key_factors': ai_result.key_factors
            })

            # Slack notification for high confidence
            if final_score >= settings.SLACK_NOTIFICATION_THRESHOLD:
                await slack_service.send_prediction_alert(prediction)
                prediction.slack_notification_sent = True

        await self.db.commit()

        logger.info(
            f"Created {len(predictions)} AI-enhanced predictions for vessel {vessel_id}"
        )

        return sorted(predictions, key=lambda p: p['confidence'], reverse=True)

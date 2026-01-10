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

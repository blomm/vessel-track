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

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

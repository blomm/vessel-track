import httpx
import logging
from typing import Optional

from src.config import settings
from src.database.models import Prediction

logger = logging.getLogger(__name__)


class SlackService:
    """
    Slack webhook integration for high-confidence prediction notifications.
    """

    async def send_prediction_alert(self, prediction: Prediction):
        """
        Send Slack notification for high-confidence prediction.

        Args:
            prediction: Prediction model with confidence >= threshold
        """
        if not settings.SLACK_WEBHOOK_URL:
            logger.info(
                f"[SLACK NOTIFICATION] Would send: {prediction.vessel.name} → "
                f"{prediction.terminal.name} ({prediction.confidence_score:.0%} confidence)"
            )
            return

        # Format message
        message = self._format_message(prediction)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.SLACK_WEBHOOK_URL,
                    json=message,
                    timeout=10.0
                )
                response.raise_for_status()

            logger.info(
                f"Slack notification sent for {prediction.vessel.name} → "
                f"{prediction.terminal.name}"
            )

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def _format_message(self, prediction: Prediction) -> dict:
        """Format Slack message payload"""
        emoji = self._get_confidence_emoji(prediction.confidence_score)

        text = (
            f"{emoji} *High Confidence Prediction*\n"
            f"*{prediction.vessel.name}* → *{prediction.terminal.name}*"
        )

        # Build attachment fields
        fields = [
            {
                "title": "Confidence",
                "value": f"{prediction.confidence_score:.0%}",
                "short": True
            },
            {
                "title": "ETA",
                "value": f"{prediction.eta_hours:.1f} hours" if prediction.eta_hours else "N/A",
                "short": True
            },
            {
                "title": "Distance",
                "value": f"{prediction.distance_to_terminal_km:.0f} km",
                "short": True
            },
            {
                "title": "Terminal Type",
                "value": prediction.terminal.terminal_type.upper(),
                "short": True
            }
        ]

        # Add AI reasoning
        if prediction.ai_reasoning:
            fields.append({
                "title": "AI Analysis",
                "value": prediction.ai_reasoning,
                "short": False
            })

        # Add score breakdown
        fields.append({
            "title": "Score Breakdown",
            "value": (
                f"• Proximity: {prediction.proximity_score:.2f}\n"
                f"• Speed: {prediction.speed_score:.2f}\n"
                f"• Heading: {prediction.heading_score:.2f}\n"
                f"• Historical: +{prediction.historical_similarity_score:.2f}\n"
                f"• AI Adjustment: {prediction.ai_confidence_adjustment:+.2f}"
            ),
            "short": False
        })

        return {
            "text": text,
            "attachments": [
                {
                    "color": self._get_confidence_color(prediction.confidence_score),
                    "fields": fields,
                    "footer": "LNG Vessel Tracker",
                    "ts": int(prediction.prediction_time.timestamp())
                }
            ]
        }

    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level"""
        if confidence >= 0.90:
            return "🎯"
        elif confidence >= 0.80:
            return "📍"
        else:
            return "🚢"

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level"""
        if confidence >= 0.90:
            return "#2eb886"  # Green
        elif confidence >= 0.80:
            return "#daa520"  # Gold
        else:
            return "#3aa3e3"  # Blue

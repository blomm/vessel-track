# Phase 3: AI Integration

**Duration**: Days 13-17
**Goal**: Integrate GPT-4o for intelligent prediction analysis and complete the prediction pipeline

---

## 3.1. AI Service with GPT-4o

### Create `service/src/services/ai_service.py`:

```python
import openai
from typing import List, Dict, Any
import json
import logging
from dataclasses import dataclass

from src.config import settings
from src.database.models import (
    Vessel,
    Terminal,
    VesselJourney,
    TerminalApproachBehavior
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


@dataclass
class AIAnalysisResult:
    """Result from GPT-4o analysis"""
    confidence_adjustment: float  # -0.3 to +0.3
    reasoning: str  # Natural language explanation
    key_factors: List[str]  # Most important factors


class AIService:
    """
    Interface with OpenAI GPT-4o for intelligent prediction analysis.
    """

    async def analyze_prediction(
        self,
        vessel: Vessel,
        terminal: Terminal,
        proximity_score: float,
        speed_score: float,
        heading_score: float,
        historical_matches: List[tuple],  # List of (VesselJourney, similarity) tuples
        approach_behaviors: List[TerminalApproachBehavior],
    ) -> AIAnalysisResult:
        """
        Send comprehensive context to GPT-4o for holistic analysis.

        Args:
            vessel: Current vessel state
            terminal: Target terminal
            proximity_score: Traditional proximity score (0-1)
            speed_score: Traditional speed score (0-1)
            heading_score: Traditional heading score (0-1)
            historical_matches: RAG-retrieved similar journeys
            approach_behaviors: Learned approach patterns

        Returns:
            AIAnalysisResult with confidence adjustment and reasoning
        """
        # Build prompt
        prompt = self._build_analysis_prompt(
            vessel=vessel,
            terminal=terminal,
            proximity_score=proximity_score,
            speed_score=speed_score,
            heading_score=heading_score,
            historical_matches=historical_matches,
            approach_behaviors=approach_behaviors,
        )

        try:
            # Call GPT-4o
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert maritime analyst specializing in "
                            "LNG tanker destination prediction. Analyze vessel "
                            "behavior and provide confidence adjustments with clear reasoning."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result_dict = json.loads(content)

            result = AIAnalysisResult(
                confidence_adjustment=float(result_dict.get("confidence_adjustment", 0.0)),
                reasoning=result_dict.get("reasoning", "No reasoning provided"),
                key_factors=result_dict.get("key_factors", [])
            )

            # Clamp adjustment to valid range
            result.confidence_adjustment = max(-0.3, min(0.3, result.confidence_adjustment))

            logger.info(
                f"AI analysis for {vessel.name} → {terminal.name}: "
                f"adjustment={result.confidence_adjustment:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Return neutral adjustment on error
            return AIAnalysisResult(
                confidence_adjustment=0.0,
                reasoning=f"AI analysis unavailable: {str(e)}",
                key_factors=[]
            )

    def _build_analysis_prompt(
        self,
        vessel: Vessel,
        terminal: Terminal,
        proximity_score: float,
        speed_score: float,
        heading_score: float,
        historical_matches: List[tuple],
        approach_behaviors: List[TerminalApproachBehavior],
    ) -> str:
        """
        Build comprehensive prompt for GPT-4o.
        """
        # Calculate distance
        from src.utils.geo import haversine_distance
        distance_km = haversine_distance(
            vessel.current_lat, vessel.current_lon,
            terminal.lat, terminal.lon
        )

        # Format historical matches
        historical_summary = self._format_historical_matches(historical_matches)

        # Format approach behaviors
        behavior_summary = self._format_approach_behaviors(approach_behaviors)

        prompt = f"""
You are analyzing a destination prediction for an LNG tanker vessel.

VESSEL DATA:
- Name: {vessel.name}
- Current Position: {vessel.current_lat:.4f}°, {vessel.current_lon:.4f}°
- Speed: {vessel.speed:.1f} knots
- Heading: {vessel.heading:.0f}°
- Vessel Type: {vessel.vessel_type}
- Status: {vessel.status or 'underway'}

TARGET TERMINAL:
- Name: {terminal.name}
- Location: {terminal.lat:.4f}°, {terminal.lon:.4f}°
- Country: {terminal.country}
- Region: {terminal.region}
- Type: {terminal.terminal_type}
- Capacity: {terminal.capacity_bcm_year:.1f} BCM/year
- Distance from Vessel: {distance_km:.1f} km

ALGORITHM SCORES (0.0-1.0):
- Proximity Score: {proximity_score:.3f}
- Speed Score: {speed_score:.3f}
- Heading Score: {heading_score:.3f}
- Base Confidence: {(proximity_score * 0.4 + speed_score * 0.3 + heading_score * 0.3):.3f}

HISTORICAL DATA (RAG Retrieved):
{historical_summary}

LEARNED APPROACH PATTERNS:
{behavior_summary}

TASK:
Analyze all factors holistically to determine if this vessel is truly heading to this terminal.

Consider:
1. Are the algorithm scores consistent with actual approach behavior?
2. Do the historical patterns support this prediction?
3. Are there any red flags (e.g., speed too high, wrong heading, terminal type mismatch)?
4. What are the most critical factors influencing confidence?

Provide:
1. confidence_adjustment: A value between -0.3 and +0.3 to adjust the base confidence score
   - Positive values (+0.1 to +0.3) if evidence strongly supports the prediction
   - Negative values (-0.1 to -0.3) if evidence contradicts the prediction
   - Near zero if neutral or uncertain
2. reasoning: A 2-3 sentence explanation of your confidence adjustment
3. key_factors: List of 2-4 most important factors that influenced your decision

Respond in JSON format:
{{
    "confidence_adjustment": <float between -0.3 and +0.3>,
    "reasoning": "<2-3 sentence explanation>",
    "key_factors": ["<factor1>", "<factor2>", ...]
}}
"""
        return prompt

    def _format_historical_matches(self, matches: List[tuple]) -> str:
        """Format historical journey matches for prompt"""
        if not matches:
            return "No similar historical journeys found."

        lines = [f"Found {len(matches)} similar completed journeys:"]

        for i, (journey, similarity) in enumerate(matches[:5], 1):
            vessel_name = journey.vessel.name if journey.vessel else "Unknown"
            origin_name = journey.origin.name if journey.origin else "Unknown"
            dest_name = journey.destination.name if journey.destination else "Unknown"

            line = f"{i}. {vessel_name}: {origin_name} → {dest_name}"

            if journey.duration_hours:
                line += f" ({journey.duration_hours:.1f}h"
            if journey.avg_speed:
                line += f", avg speed {journey.avg_speed:.1f}kt"
            if journey.duration_hours:
                line += ")"

            line += f" [similarity: {similarity:.3f}]"
            lines.append(line)

        return "\n".join(lines)

    def _format_approach_behaviors(
        self,
        behaviors: List[TerminalApproachBehavior]
    ) -> str:
        """Format learned approach behaviors for prompt"""
        if not behaviors:
            return "No learned approach patterns available for this terminal."

        lines = ["Typical approach patterns for this terminal:"]

        for i, behavior in enumerate(behaviors[:3], 1):
            vessel_specific = " (vessel-specific)" if behavior.vessel_id else " (general)"

            line = f"{i}. Distance: {behavior.approach_distance_km:.1f}km{vessel_specific}"

            if behavior.typical_speed_range_min and behavior.typical_speed_range_max:
                line += (
                    f", Speed: {behavior.typical_speed_range_min:.1f}-"
                    f"{behavior.typical_speed_range_max:.1f}kt"
                )

            if behavior.observation_count:
                line += f", Observed: {behavior.observation_count} times"

            if behavior.confidence:
                line += f", Confidence: {behavior.confidence:.2f}"

            lines.append(line)

        return "\n".join(lines)
```

---

## 3.2. Complete Prediction Pipeline

### Update `service/src/services/prediction_engine.py`:

Add the following method to integrate AI and RAG:

```python
from src.services.ai_service import AIService
from src.services.rag_service import RAGService

class PredictionEngine:
    """Enhanced prediction engine with AI and RAG integration"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.ai_service = AIService()
        self.rag_service = RAGService(db_session)

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
        """
        from src.database.models import Prediction as PredictionModel
        from datetime import datetime, timedelta
        from src.services.slack_service import SlackService

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
            from src.utils.geo import haversine_distance
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
            prediction = PredictionModel(
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
```

---

## 3.3. Slack Service (Placeholder)

### Create `service/src/services/slack_service.py`:

```python
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
```

---

## Verification Checklist

- [ ] `AIService` implemented with OpenAI GPT-4o client
- [ ] Prompt engineering complete (comprehensive context)
- [ ] JSON response parsing works
- [ ] Confidence adjustment clamped to -0.3 to +0.3
- [ ] `PredictionEngine.analyze_vessel_with_ai()` implemented
- [ ] All services integrated (traditional + RAG + AI)
- [ ] Predictions stored in database with all scores
- [ ] `SlackService` implemented with webhook
- [ ] Slack notifications triggered for confidence >= 80%
- [ ] Can run end-to-end prediction with real vessel data
- [ ] AI reasoning appears in prediction results
- [ ] Key factors tracked

---

## Testing

### Manual Test:

```bash
cd service

# Test AI analysis
poetry run python -c "
import asyncio
from src.database.connection import AsyncSessionLocal
from src.services.prediction_engine import PredictionEngine

async def test():
    async with AsyncSessionLocal() as session:
        engine = PredictionEngine(session)
        predictions = await engine.analyze_vessel_with_ai('lng-001')
        for p in predictions[:3]:
            print(f\"Terminal: {p['terminal_name']}\")
            print(f\"Confidence: {p['confidence']:.2%}\")
            print(f\"AI Reasoning: {p['ai_reasoning']}\")
            print(f\"Key Factors: {', '.join(p['key_factors'])}\")
            print('---')

asyncio.run(test())
"
```

---

## Next Steps

Once this phase is complete, move to **Phase 4: Learning System** where we'll implement:
- Learning service for processing prediction outcomes
- Accuracy calculation
- Approach behavior updates
- Behavior event detection
- Feedback loop for continuous improvement

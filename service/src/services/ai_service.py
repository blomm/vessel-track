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
from src.utils.geo import haversine_distance

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

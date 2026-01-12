import asyncio
from datetime import datetime
from src.database.connection import AsyncSessionLocal
from src.database.models import Vessel
from src.services.prediction_engine import PredictionEngine

async def create_test_vessel():
    """Create a test vessel in the Gulf of Mexico heading toward Sabine Pass"""
    async with AsyncSessionLocal() as session:
        # Check if vessel already exists
        existing = await session.get(Vessel, "lng-test-001")
        if existing:
            print(f"Test vessel already exists: {existing.name}")
            return existing

        # Create test vessel positioned near Sabine Pass LNG terminal
        # Sabine Pass is at: 29.7294, -93.8767
        # Position vessel 100km south, heading north (0 degrees)
        vessel = Vessel(
            id="lng-test-001",
            name="Pacific Spirit",
            current_lat=28.8,  # ~100km south of Sabine Pass
            current_lon=-93.8767,
            heading=0.0,  # North
            speed=12.0,  # 12 knots - optimal approach speed
            vessel_type="lng_tanker",
            mmsi="123456789",
            imo="IMO1234567",
            status="underway",
            last_updated=datetime.utcnow()
        )
        session.add(vessel)
        await session.commit()
        print(f"Created test vessel: {vessel.name}")
        return vessel


async def test_ai_prediction_engine():
    """Test the AI-enhanced prediction engine"""
    print("\n" + "="*70)
    print("Testing AI-Enhanced Prediction Engine (Phase 3)")
    print("="*70 + "\n")

    # Create test vessel
    vessel = await create_test_vessel()

    # Run AI-enhanced prediction
    async with AsyncSessionLocal() as session:
        engine = PredictionEngine(session)
        print(f"Running AI-enhanced predictions for {vessel.name}...\n")
        predictions = await engine.analyze_vessel_with_ai(vessel.id)

        print(f"Generated {len(predictions)} AI-enhanced predictions:\n")
        print("-" * 70)

        for i, pred in enumerate(predictions, 1):
            print(f"\n{i}. Terminal: {pred['terminal_name']}")
            print(f"   Confidence: {pred['confidence']:.2%}")
            if pred['eta_hours']:
                print(f"   ETA: {pred['eta_hours']:.1f} hours")

            print(f"\n   AI Reasoning:")
            print(f"   {pred['ai_reasoning']}")

            if pred['key_factors']:
                print(f"\n   Key Factors:")
                for factor in pred['key_factors']:
                    print(f"   • {factor}")

            print("-" * 70)

    print("\n" + "="*70)
    print("Test Complete - Check database for prediction records")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_ai_prediction_engine())

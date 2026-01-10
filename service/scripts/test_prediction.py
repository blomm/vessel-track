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


async def test_prediction_engine():
    """Test the prediction engine with a sample vessel"""
    print("\n=== Testing Prediction Engine ===\n")

    # Create test vessel
    vessel = await create_test_vessel()

    # Run prediction
    async with AsyncSessionLocal() as session:
        engine = PredictionEngine(session)
        predictions = await engine.analyze_vessel(vessel.id)

        print(f"\nGenerated {len(predictions)} predictions for {vessel.name}:\n")

        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred.terminal_name}")
            print(f"   Confidence: {pred.confidence_score:.2%}")
            print(f"   Distance: {pred.distance_km:.1f} km")
            if pred.eta_hours:
                print(f"   ETA: {pred.eta_hours:.1f} hours")
            print(f"   Scores - Proximity: {pred.proximity_score:.2f}, "
                  f"Speed: {pred.speed_score:.2f}, Heading: {pred.heading_score:.2f}")
            print()

    print("=== Test Complete ===\n")


if __name__ == "__main__":
    asyncio.run(test_prediction_engine())

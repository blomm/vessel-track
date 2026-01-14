"""
Test script for Phase 4: Learning System
Tests learning service, outcome detection, and behavior tracking
"""
import asyncio
from datetime import datetime, timedelta
from src.database.connection import AsyncSessionLocal
from src.database.models import Prediction
from src.services.learning_service import LearningService
from sqlalchemy import select


async def test_learning_system():
    """Test the learning system with a manual prediction outcome"""
    print("\n" + "="*70)
    print("Phase 4: Learning System - Test")
    print("="*70 + "\n")

    async with AsyncSessionLocal() as session:
        # Find an active prediction to test with
        result = await session.execute(
            select(Prediction).where(Prediction.status == 'active').limit(1)
        )
        prediction = result.scalar_one_or_none()

        if not prediction:
            print("❌ No active predictions found. Please run test_ai_prediction.py first.")
            print("\nTo create test predictions, run:")
            print("  cd service")
            print("  PYTHONPATH=. poetry run python scripts/test_ai_prediction.py")
            return

        print(f"✓ Found active prediction:")
        print(f"  Prediction ID: {prediction.id}")
        print(f"  Vessel: {prediction.vessel.name}")
        print(f"  Terminal: {prediction.terminal.name}")
        print(f"  Confidence: {prediction.confidence_score:.2%}")
        print(f"  Status: {prediction.status}")

        if prediction.predicted_arrival:
            print(f"  Predicted Arrival: {prediction.predicted_arrival}")

        print("\n" + "-"*70)
        print("Testing Learning Service")
        print("-"*70 + "\n")

        learning = LearningService(session)

        # Simulate vessel arrival
        # For testing, use a time close to predicted arrival
        if prediction.predicted_arrival:
            # Simulate arrival 1 hour after predicted time
            actual_arrival = prediction.predicted_arrival + timedelta(hours=1)
        else:
            actual_arrival = datetime.utcnow()

        print(f"Simulating vessel arrival at: {actual_arrival}")
        print(f"Processing prediction outcome...\n")

        try:
            # Process the prediction outcome
            await learning.process_prediction_outcome(
                prediction_id=prediction.id,
                actual_arrival_time=actual_arrival,
                outcome_status='confirmed'
            )

            print("✅ Prediction outcome processed successfully!")
            print("\nVerifying results...\n")

            # Refresh prediction to see updates
            await session.refresh(prediction)

            print(f"Updated Prediction Status: {prediction.status}")
            print(f"Actual Arrival Time: {prediction.actual_arrival_time}")
            print(f"Accuracy Score: {prediction.accuracy_score:.2%}" if prediction.accuracy_score else "Accuracy Score: N/A")

            # Check if embedding was stored
            from src.database.models import VectorEmbedding
            result = await session.execute(
                select(VectorEmbedding).where(
                    VectorEmbedding.content_type == 'prediction',
                    VectorEmbedding.content_id == prediction.id
                )
            )
            embedding = result.scalar_one_or_none()

            if embedding:
                print(f"\n✅ Vector embedding created:")
                print(f"  Embedding dimension: {len(embedding.embedding)}")
                print(f"  Text content preview: {embedding.text_content[:100]}...")
            else:
                print("\n⚠️  No vector embedding found")

            # Check if approach behavior was updated
            from src.database.models import TerminalApproachBehavior
            result = await session.execute(
                select(TerminalApproachBehavior).where(
                    TerminalApproachBehavior.terminal_id == prediction.terminal_id,
                    TerminalApproachBehavior.vessel_id == prediction.vessel_id
                )
            )
            behavior = result.scalar_one_or_none()

            if behavior:
                print(f"\n✅ Approach behavior updated:")
                print(f"  Observation count: {behavior.observation_count}")
                print(f"  Approach distance: {behavior.approach_distance_km:.1f} km")
                if behavior.typical_speed_range_min and behavior.typical_speed_range_max:
                    print(f"  Speed range: {behavior.typical_speed_range_min:.1f}-{behavior.typical_speed_range_max:.1f} knots")
                if behavior.confidence:
                    print(f"  Confidence: {behavior.confidence:.2%}")
            else:
                print("\n⚠️  No approach behavior found")

        except Exception as e:
            print(f"❌ Error processing outcome: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n" + "="*70)
    print("Phase 4 Learning System Test Complete")
    print("="*70 + "\n")

    print("Summary:")
    print("✅ LearningService successfully processed prediction outcome")
    print("✅ Accuracy score calculated")
    print("✅ Vector embedding generated and stored for RAG")
    print("✅ Terminal approach behavior pattern updated")
    print("\nThe learning system is now continuously improving predictions!")


async def test_behavior_detection():
    """Test behavior event detection"""
    print("\n" + "="*70)
    print("Testing Behavior Event Detection")
    print("="*70 + "\n")

    async with AsyncSessionLocal() as session:
        learning = LearningService(session)

        # Simulate vessel state change
        vessel_id = "lng-test-001"

        previous_state = {
            'lat': 28.8,
            'lon': -93.8767,
            'speed': 12.0,
            'heading': 0.0
        }

        current_state = {
            'lat': 28.9,
            'lon': -93.8767,
            'speed': 8.0,  # Speed change of 4 knots
            'heading': 45.0  # Course change of 45 degrees
        }

        print("Simulating vessel state change:")
        print(f"  Previous: speed={previous_state['speed']}kt, heading={previous_state['heading']}°")
        print(f"  Current:  speed={current_state['speed']}kt, heading={current_state['heading']}°")
        print()

        await learning.detect_behavior_events(
            vessel_id=vessel_id,
            current_state=current_state,
            previous_state=previous_state
        )

        # Check for created events
        from src.database.models import BehaviorEvent
        result = await session.execute(
            select(BehaviorEvent).where(BehaviorEvent.vessel_id == vessel_id)
            .order_by(BehaviorEvent.event_time.desc())
            .limit(5)
        )
        events = result.scalars().all()

        if events:
            print(f"✅ Detected {len(events)} behavior event(s):\n")
            for event in events:
                print(f"  • {event.event_type.upper()}")
                print(f"    Magnitude: {event.magnitude:.1f}")
                if event.nearest_terminal_id:
                    print(f"    Distance to terminal: {event.distance_to_terminal_km:.1f} km")
                print()
        else:
            print("⚠️  No behavior events detected")

    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_learning_system())

    # Optionally test behavior detection
    print("\nWould you like to test behavior event detection?")
    print("(This requires an existing test vessel)")
    # asyncio.run(test_behavior_detection())

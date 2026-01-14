"""
Test script to verify Phase 4 code structure without requiring API calls
"""
import asyncio
from datetime import datetime
from src.database.connection import AsyncSessionLocal


async def verify_phase4_implementation():
    """Verify that all Phase 4 components are correctly implemented"""
    print("\n" + "="*70)
    print("Phase 4: Learning System - Structure Verification")
    print("="*70 + "\n")

    checks_passed = []
    checks_failed = []

    # Check 1: LearningService exists and has correct methods
    print("✓ Checking LearningService implementation...")
    try:
        from src.services.learning_service import LearningService

        async with AsyncSessionLocal() as session:
            learning_service = LearningService(session)

            assert hasattr(learning_service, 'process_prediction_outcome'), "Missing process_prediction_outcome method"
            assert hasattr(learning_service, 'detect_behavior_events'), "Missing detect_behavior_events method"
            assert hasattr(learning_service, '_calculate_accuracy'), "Missing _calculate_accuracy method"
            assert hasattr(learning_service, '_store_prediction_embedding'), "Missing _store_prediction_embedding method"
            assert hasattr(learning_service, '_update_approach_behaviors'), "Missing _update_approach_behaviors method"
            assert hasattr(learning_service, '_create_behavior_event'), "Missing _create_behavior_event method"
            assert hasattr(learning_service, 'embedding_service'), "Missing embedding_service attribute"

        checks_passed.append("LearningService structure")
        print("  ✅ LearningService has all required methods")
        print("  ✅ LearningService has embedding_service instance")
    except Exception as e:
        checks_failed.append(f"LearningService: {e}")
        print(f"  ❌ LearningService check failed: {e}")

    # Check 2: OutcomeDetector exists and has correct methods
    print("\n✓ Checking OutcomeDetector implementation...")
    try:
        from src.services.outcome_detector import OutcomeDetector

        async with AsyncSessionLocal() as session:
            detector = OutcomeDetector(session)

            assert hasattr(detector, 'check_active_predictions'), "Missing check_active_predictions method"
            assert hasattr(detector, '_check_prediction'), "Missing _check_prediction method"
            assert hasattr(detector, 'learning_service'), "Missing learning_service attribute"

        checks_passed.append("OutcomeDetector structure")
        print("  ✅ OutcomeDetector has all required methods")
        print("  ✅ OutcomeDetector has learning_service instance")
    except Exception as e:
        checks_failed.append(f"OutcomeDetector: {e}")
        print(f"  ❌ OutcomeDetector check failed: {e}")

    # Check 3: Background tasks module exists
    print("\n✓ Checking background tasks implementation...")
    try:
        from src.services.background_tasks import run_outcome_detection, run_all_background_tasks

        assert callable(run_outcome_detection), "run_outcome_detection is not callable"
        assert callable(run_all_background_tasks), "run_all_background_tasks is not callable"

        checks_passed.append("Background tasks module")
        print("  ✅ Background tasks module has required functions")
    except Exception as e:
        checks_failed.append(f"Background tasks: {e}")
        print(f"  ❌ Background tasks check failed: {e}")

    # Check 4: Main.py exists with startup/shutdown hooks
    print("\n✓ Checking main.py FastAPI integration...")
    try:
        from src.main import app, startup_event, shutdown_event

        assert app is not None, "FastAPI app not initialized"
        assert callable(startup_event), "startup_event is not callable"
        assert callable(shutdown_event), "shutdown_event is not callable"

        checks_passed.append("FastAPI integration")
        print("  ✅ main.py has FastAPI app")
        print("  ✅ main.py has startup/shutdown event handlers")
    except Exception as e:
        checks_failed.append(f"FastAPI integration: {e}")
        print(f"  ❌ FastAPI integration check failed: {e}")

    # Check 5: Database models have required fields
    print("\n✓ Checking database model fields...")
    try:
        from src.database.models import Prediction, BehaviorEvent, TerminalApproachBehavior

        # Prediction fields for learning
        prediction_fields = [
            'actual_arrival_time',
            'accuracy_score',
            'status'
        ]
        for field in prediction_fields:
            assert hasattr(Prediction, field), f"Missing Prediction field: {field}"

        # BehaviorEvent fields
        behavior_event_fields = [
            'vessel_id',
            'event_type',
            'lat',
            'lon',
            'speed_before',
            'speed_after',
            'heading_before',
            'heading_after',
            'magnitude',
            'nearest_terminal_id',
            'distance_to_terminal_km'
        ]
        for field in behavior_event_fields:
            assert hasattr(BehaviorEvent, field), f"Missing BehaviorEvent field: {field}"

        # TerminalApproachBehavior fields
        approach_fields = [
            'terminal_id',
            'vessel_id',
            'approach_distance_km',
            'typical_speed_range_min',
            'typical_speed_range_max',
            'observation_count',
            'confidence',
            'last_observed'
        ]
        for field in approach_fields:
            assert hasattr(TerminalApproachBehavior, field), f"Missing TerminalApproachBehavior field: {field}"

        checks_passed.append("Database models")
        print(f"  ✅ Prediction model has all {len(prediction_fields)} learning fields")
        print(f"  ✅ BehaviorEvent model has all {len(behavior_event_fields)} fields")
        print(f"  ✅ TerminalApproachBehavior model has all {len(approach_fields)} fields")
    except Exception as e:
        checks_failed.append(f"Database models: {e}")
        print(f"  ❌ Database models check failed: {e}")

    # Check 6: EmbeddingService has prediction outcome support
    print("\n✓ Checking EmbeddingService prediction outcome support...")
    try:
        from src.services.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()
        assert hasattr(embedding_service, 'embed_prediction_outcome'), "Missing embed_prediction_outcome method"
        assert hasattr(embedding_service, '_prediction_to_text'), "Missing _prediction_to_text method"

        checks_passed.append("EmbeddingService prediction support")
        print("  ✅ EmbeddingService has embed_prediction_outcome method")
        print("  ✅ EmbeddingService has _prediction_to_text method")
    except Exception as e:
        checks_failed.append(f"EmbeddingService: {e}")
        print(f"  ❌ EmbeddingService check failed: {e}")

    # Check 7: Verify geo utils has angular_difference
    print("\n✓ Checking geo utilities...")
    try:
        from src.utils.geo import haversine_distance, angular_difference

        assert callable(haversine_distance), "haversine_distance is not callable"
        assert callable(angular_difference), "angular_difference is not callable"

        checks_passed.append("Geo utilities")
        print("  ✅ haversine_distance available")
        print("  ✅ angular_difference available")
    except Exception as e:
        checks_failed.append(f"Geo utilities: {e}")
        print(f"  ❌ Geo utilities check failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70 + "\n")
    print(f"✅ Checks Passed: {len(checks_passed)}")
    for check in checks_passed:
        print(f"   • {check}")

    if checks_failed:
        print(f"\n❌ Checks Failed: {len(checks_failed)}")
        for check in checks_failed:
            print(f"   • {check}")

    print("\n" + "="*70)
    if not checks_failed:
        print("✅ Phase 4 Implementation: COMPLETE")
        print("\nAll components are correctly structured and ready for use.")
        print("\nNext steps:")
        print("  1. Run test_phase4_learning.py to test learning system")
        print("  2. Background tasks will automatically run when FastAPI starts")
        print("  3. Outcome detection runs every 5 minutes")
        print("\nNote: Full testing requires:")
        print("  • Active predictions in database")
        print("  • OpenAI API key with available quota (for embeddings)")
    else:
        print("❌ Phase 4 Implementation: INCOMPLETE")
        print("\nPlease fix the failed checks above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(verify_phase4_implementation())

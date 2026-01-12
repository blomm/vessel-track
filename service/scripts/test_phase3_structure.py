"""
Test script to verify Phase 3 code structure without requiring OpenAI API calls
"""
import asyncio
from datetime import datetime
from src.database.connection import AsyncSessionLocal
from src.database.models import Vessel, Terminal
from src.services.ai_service import AIService
from src.services.slack_service import SlackService
from src.services.prediction_engine import PredictionEngine

async def verify_phase3_implementation():
    """Verify that all Phase 3 components are correctly implemented"""
    print("\n" + "="*70)
    print("Phase 3: AI Integration - Structure Verification")
    print("="*70 + "\n")

    checks_passed = []
    checks_failed = []

    # Check 1: AI Service exists and has correct methods
    print("✓ Checking AIService implementation...")
    try:
        ai_service = AIService()
        assert hasattr(ai_service, 'analyze_prediction'), "Missing analyze_prediction method"
        assert hasattr(ai_service, '_build_analysis_prompt'), "Missing _build_analysis_prompt method"
        assert hasattr(ai_service, '_format_historical_matches'), "Missing _format_historical_matches method"
        assert hasattr(ai_service, '_format_approach_behaviors'), "Missing _format_approach_behaviors method"
        checks_passed.append("AIService structure")
        print("  ✅ AIService has all required methods")
    except Exception as e:
        checks_failed.append(f"AIService: {e}")
        print(f"  ❌ AIService check failed: {e}")

    # Check 2: Slack Service exists and has correct methods
    print("\n✓ Checking SlackService implementation...")
    try:
        slack_service = SlackService()
        assert hasattr(slack_service, 'send_prediction_alert'), "Missing send_prediction_alert method"
        assert hasattr(slack_service, '_format_message'), "Missing _format_message method"
        assert hasattr(slack_service, '_get_confidence_emoji'), "Missing _get_confidence_emoji method"
        assert hasattr(slack_service, '_get_confidence_color'), "Missing _get_confidence_color method"
        checks_passed.append("SlackService structure")
        print("  ✅ SlackService has all required methods")
    except Exception as e:
        checks_failed.append(f"SlackService: {e}")
        print(f"  ❌ SlackService check failed: {e}")

    # Check 3: PredictionEngine has new AI method
    print("\n✓ Checking PredictionEngine AI integration...")
    try:
        async with AsyncSessionLocal() as session:
            engine = PredictionEngine(session)
            assert hasattr(engine, 'analyze_vessel_with_ai'), "Missing analyze_vessel_with_ai method"
            assert hasattr(engine, 'ai_service'), "Missing ai_service attribute"
            assert hasattr(engine, 'rag_service'), "Missing rag_service attribute"
            checks_passed.append("PredictionEngine AI integration")
            print("  ✅ PredictionEngine has analyze_vessel_with_ai method")
            print("  ✅ PredictionEngine has ai_service instance")
            print("  ✅ PredictionEngine has rag_service instance")
    except Exception as e:
        checks_failed.append(f"PredictionEngine: {e}")
        print(f"  ❌ PredictionEngine check failed: {e}")

    # Check 4: Database models have required fields
    print("\n✓ Checking database Prediction model fields...")
    try:
        from src.database.models import Prediction
        required_fields = [
            'ai_confidence_adjustment',
            'ai_reasoning',
            'proximity_score',
            'speed_score',
            'heading_score',
            'historical_similarity_score',
            'slack_notification_sent'
        ]
        for field in required_fields:
            assert hasattr(Prediction, field), f"Missing field: {field}"
        checks_passed.append("Database Prediction model")
        print(f"  ✅ All {len(required_fields)} required fields present in Prediction model")
    except Exception as e:
        checks_failed.append(f"Database model: {e}")
        print(f"  ❌ Database model check failed: {e}")

    # Check 5: Test vessel exists in database
    print("\n✓ Checking database connectivity and test data...")
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, func

            # Count vessels
            result = await session.execute(select(func.count()).select_from(Vessel))
            vessel_count = result.scalar()

            # Count terminals
            result = await session.execute(select(func.count()).select_from(Terminal))
            terminal_count = result.scalar()

            checks_passed.append("Database connectivity")
            print(f"  ✅ Database connected successfully")
            print(f"  ✅ Found {vessel_count} vessel(s) in database")
            print(f"  ✅ Found {terminal_count} terminal(s) in database")
    except Exception as e:
        checks_failed.append(f"Database: {e}")
        print(f"  ❌ Database check failed: {e}")

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
        print("✅ Phase 3 Implementation: COMPLETE")
        print("\nAll components are correctly structured and ready for use.")
        print("\nNote: Full end-to-end testing requires:")
        print("  • Valid OpenAI API key with available quota")
        print("  • Historical journey data in database (for RAG)")
    else:
        print("❌ Phase 3 Implementation: INCOMPLETE")
        print("\nPlease fix the failed checks above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(verify_phase3_implementation())

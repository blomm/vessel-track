"""
Test script for Phase 5: API & WebSocket
Verifies all API endpoints are correctly configured
"""
import asyncio
from src.main import app


def test_api_structure():
    """Test that all API routes are properly configured"""
    print("\n" + "="*70)
    print("Phase 5: API & WebSocket - Structure Verification")
    print("="*70 + "\n")

    checks_passed = []
    checks_failed = []

    # Check 1: App configuration
    print("✓ Checking FastAPI app configuration...")
    try:
        assert app.title == "Vessel Track API"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
        checks_passed.append("FastAPI app configuration")
        print("  ✅ App title, docs, and redoc configured")
    except Exception as e:
        checks_failed.append(f"FastAPI configuration: {e}")
        print(f"  ❌ FastAPI configuration failed: {e}")

    # Check 2: CORS middleware
    print("\n✓ Checking CORS middleware...")
    try:
        has_cors = any(
            "CORSMiddleware" in str(middleware)
            for middleware in app.user_middleware
        )
        assert has_cors, "CORS middleware not found"
        assert len(app.user_middleware) > 0, "No middleware configured"
        checks_passed.append("CORS middleware")
        print("  ✅ CORS middleware configured")
    except Exception as e:
        checks_failed.append(f"CORS middleware: {e}")
        print(f"  ❌ CORS middleware check failed: {e}")

    # Check 3: Routes
    print("\n✓ Checking API routes...")
    routes = [route for route in app.routes if hasattr(route, "path")]
    route_paths = [route.path for route in routes]

    expected_prefixes = [
        "/api/v1/vessels",
        "/api/v1/terminals",
        "/api/v1/predictions",
        "/api/v1/admin",
        "/ws/vessels"
    ]

    try:
        for prefix in expected_prefixes:
            matching_routes = [path for path in route_paths if path.startswith(prefix)]
            assert len(matching_routes) > 0, f"No routes found for prefix {prefix}"
            print(f"  ✅ {prefix} - {len(matching_routes)} route(s)")

        checks_passed.append("API routes")
    except Exception as e:
        checks_failed.append(f"API routes: {e}")
        print(f"  ❌ API routes check failed: {e}")

    # Check 4: Specific endpoints
    print("\n✓ Checking specific endpoints...")
    required_endpoints = {
        "/": "Root",
        "/health": "Health check",
        "/api/v1/vessels": "List vessels",
        "/api/v1/terminals": "List terminals",
        "/api/v1/predictions/analyze": "Analyze vessel",
        "/api/v1/predictions/active": "Active predictions",
        "/api/v1/admin/health": "Admin health",
        "/api/v1/admin/metrics": "Admin metrics",
        "/api/v1/admin/accuracy": "Admin accuracy",
        "/ws/vessels": "WebSocket"
    }

    try:
        for path, name in required_endpoints.items():
            if path in route_paths or any(rp.startswith(path) for rp in route_paths):
                print(f"  ✅ {name}: {path}")
            else:
                raise AssertionError(f"Missing endpoint: {name} ({path})")

        checks_passed.append("Required endpoints")
    except Exception as e:
        checks_failed.append(f"Required endpoints: {e}")
        print(f"  ❌ Required endpoints check failed: {e}")

    # Check 5: Routers imported
    print("\n✓ Checking router imports...")
    try:
        from src.api.routers import vessels, terminals, predictions, admin, websocket

        assert hasattr(vessels, 'router'), "Vessels router not found"
        assert hasattr(terminals, 'router'), "Terminals router not found"
        assert hasattr(predictions, 'router'), "Predictions router not found"
        assert hasattr(admin, 'router'), "Admin router not found"
        assert hasattr(websocket, 'router'), "WebSocket router not found"

        checks_passed.append("Router imports")
        print("  ✅ All routers successfully imported")
    except Exception as e:
        checks_failed.append(f"Router imports: {e}")
        print(f"  ❌ Router imports check failed: {e}")

    # Check 6: Schemas
    print("\n✓ Checking Pydantic schemas...")
    try:
        from src.schemas.vessel import VesselResponse, VesselCreate, VesselUpdate
        from src.schemas.terminal import TerminalResponse, TerminalCreate
        from src.schemas.prediction import PredictionDetail, AnalyzeRequest

        checks_passed.append("Pydantic schemas")
        print("  ✅ All schemas successfully imported")
    except Exception as e:
        checks_failed.append(f"Pydantic schemas: {e}")
        print(f"  ❌ Pydantic schemas check failed: {e}")

    # Check 7: WebSocket manager
    print("\n✓ Checking WebSocket manager...")
    try:
        from src.services.websocket_manager import manager

        assert hasattr(manager, 'connect'), "Missing connect method"
        assert hasattr(manager, 'disconnect'), "Missing disconnect method"
        assert hasattr(manager, 'subscribe'), "Missing subscribe method"
        assert hasattr(manager, 'broadcast_vessel_update'), "Missing broadcast_vessel_update method"

        checks_passed.append("WebSocket manager")
        print("  ✅ WebSocket manager has all required methods")
    except Exception as e:
        checks_failed.append(f"WebSocket manager: {e}")
        print(f"  ❌ WebSocket manager check failed: {e}")

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
        print("✅ Phase 5 Implementation: COMPLETE")
        print("\nAll API components are correctly structured and ready for use.")
        print("\nTo start the server:")
        print("  cd service")
        print("  poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
        print("\nThen visit:")
        print("  • http://localhost:8000/docs - Swagger UI")
        print("  • http://localhost:8000/redoc - ReDoc")
        print("  • http://localhost:8000/api/v1/vessels - List vessels")
        print("  • http://localhost:8000/api/v1/admin/metrics - System metrics")
    else:
        print("❌ Phase 5 Implementation: INCOMPLETE")
        print("\nPlease fix the failed checks above.")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_api_structure()

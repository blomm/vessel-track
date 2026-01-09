# Phase 5: API & WebSocket

**Duration**: Days 22-26
**Goal**: Implement REST API endpoints, WebSocket for real-time updates, and admin functionality

---

## 5.1. FastAPI Main Application

### Create `service/src/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from src.config import settings
from src.api.routers import vessels, terminals, predictions, admin, websocket
from src.services.background_tasks import run_all_background_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered LNG vessel destination prediction service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
        "http://localhost:3001",  # Alternative port
        # Add production domains here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vessels.router, prefix=f"{settings.API_V1_PREFIX}/vessels", tags=["Vessels"])
app.include_router(terminals.router, prefix=f"{settings.API_V1_PREFIX}/terminals", tags=["Terminals"])
app.include_router(predictions.router, prefix=f"{settings.API_V1_PREFIX}/predictions", tags=["Predictions"])
app.include_router(admin.router, prefix=f"{settings.API_V1_PREFIX}/admin", tags=["Admin"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# Background task handle
background_task = None

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global background_task
    background_task = asyncio.create_task(run_all_background_tasks())
    logger.info("🚀 Vessel Track API started")
    logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"📡 WebSocket endpoint: /ws/vessels")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks"""
    global background_task
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            logger.info("Background tasks stopped")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/admin/health"
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}
```

---

## 5.2. Pydantic Schemas

### Create `service/src/schemas/vessel.py`:

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class VesselBase(BaseModel):
    name: str
    current_lat: float
    current_lon: float
    heading: Optional[float] = None
    speed: Optional[float] = None
    vessel_type: str = 'lng_tanker'

class VesselCreate(VesselBase):
    id: str
    mmsi: Optional[str] = None
    imo: Optional[str] = None

class VesselUpdate(BaseModel):
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
    status: Optional[str] = None

class VesselResponse(VesselBase):
    id: str
    mmsi: Optional[str]
    imo: Optional[str]
    status: Optional[str]
    last_updated: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class VesselWithPredictions(VesselResponse):
    active_predictions: List['PredictionSummary'] = []
```

### Create `service/src/schemas/terminal.py`:

```python
from pydantic import BaseModel
from typing import Optional

class TerminalBase(BaseModel):
    name: str
    code: str
    lat: float
    lon: float
    country: str
    region: str
    terminal_type: str
    capacity_bcm_year: Optional[float] = None
    approach_zone_radius_km: float = 50.0

class TerminalCreate(TerminalBase):
    pass

class TerminalResponse(TerminalBase):
    id: int

    class Config:
        from_attributes = True
```

### Create `service/src/schemas/prediction.py`:

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class PredictionSummary(BaseModel):
    id: int
    terminal_id: int
    terminal_name: str
    confidence_score: float
    distance_to_terminal_km: float
    eta_hours: Optional[float]
    status: str

class PredictionDetail(PredictionSummary):
    vessel_id: str
    vessel_name: str
    predicted_arrival: Optional[datetime]
    proximity_score: float
    speed_score: float
    heading_score: float
    historical_similarity_score: float
    ai_confidence_adjustment: float
    ai_reasoning: str
    prediction_time: datetime

    class Config:
        from_attributes = True

class AnalyzeRequest(BaseModel):
    vessel_id: str

class ConfirmOutcomeRequest(BaseModel):
    actual_arrival_time: Optional[datetime] = None
    status: str = 'confirmed'  # or 'incorrect'
```

---

## 5.3. Vessel Router

### Create `service/src/api/routers/vessels.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from src.database.connection import get_db
from src.database.models import Vessel, Prediction
from src.schemas.vessel import VesselResponse, VesselCreate, VesselUpdate, VesselWithPredictions
from src.schemas.prediction import PredictionSummary

router = APIRouter()

@router.get("", response_model=List[VesselResponse])
async def list_vessels(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """List all vessels with pagination"""
    result = await db.execute(
        select(Vessel).offset(skip).limit(limit)
    )
    vessels = result.scalars().all()
    return vessels

@router.get("/{vessel_id}", response_model=VesselWithPredictions)
async def get_vessel(
    vessel_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get specific vessel with active predictions"""
    result = await db.execute(
        select(Vessel).where(Vessel.id == vessel_id)
    )
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")

    # Get active predictions
    pred_result = await db.execute(
        select(Prediction).where(
            Prediction.vessel_id == vessel_id,
            Prediction.status == 'active'
        )
    )
    predictions = pred_result.scalars().all()

    # Build response
    vessel_dict = VesselResponse.from_orm(vessel).dict()
    vessel_dict['active_predictions'] = [
        PredictionSummary(
            id=p.id,
            terminal_id=p.terminal_id,
            terminal_name=p.terminal.name,
            confidence_score=p.confidence_score,
            distance_to_terminal_km=p.distance_to_terminal_km,
            eta_hours=p.eta_hours,
            status=p.status
        ) for p in predictions
    ]

    return vessel_dict

@router.post("", response_model=VesselResponse)
async def create_vessel(
    vessel: VesselCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create new vessel"""
    db_vessel = Vessel(**vessel.dict())
    db.add(db_vessel)
    await db.commit()
    await db.refresh(db_vessel)
    return db_vessel

@router.put("/{vessel_id}", response_model=VesselResponse)
async def update_vessel(
    vessel_id: str,
    vessel_update: VesselUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update vessel data"""
    result = await db.execute(
        select(Vessel).where(Vessel.id == vessel_id)
    )
    vessel = result.scalar_one_or_none()

    if not vessel:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")

    # Update fields
    for field, value in vessel_update.dict(exclude_unset=True).items():
        setattr(vessel, field, value)

    await db.commit()
    await db.refresh(vessel)
    return vessel
```

---

## 5.4. Predictions Router

### Create `service/src/api/routers/predictions.py`:

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from src.database.connection import get_db
from src.database.models import Prediction
from src.schemas.prediction import (
    PredictionDetail,
    AnalyzeRequest,
    ConfirmOutcomeRequest
)
from src.services.prediction_engine import PredictionEngine
from src.services.learning_service import LearningService

router = APIRouter()

@router.post("/analyze")
async def analyze_vessel(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger prediction analysis for a vessel.
    Runs full pipeline: traditional + RAG + AI.
    """
    engine = PredictionEngine(db)
    predictions = await engine.analyze_vessel_with_ai(request.vessel_id)
    return {
        "vessel_id": request.vessel_id,
        "predictions_created": len(predictions),
        "predictions": predictions
    }

@router.get("/active", response_model=List[PredictionDetail])
async def list_active_predictions(
    db: AsyncSession = Depends(get_db)
):
    """Get all active predictions"""
    result = await db.execute(
        select(Prediction).where(Prediction.status == 'active')
    )
    predictions = result.scalars().all()

    return [
        PredictionDetail(
            id=p.id,
            vessel_id=p.vessel_id,
            vessel_name=p.vessel.name,
            terminal_id=p.terminal_id,
            terminal_name=p.terminal.name,
            confidence_score=p.confidence_score,
            distance_to_terminal_km=p.distance_to_terminal_km,
            eta_hours=p.eta_hours,
            predicted_arrival=p.predicted_arrival,
            proximity_score=p.proximity_score,
            speed_score=p.speed_score,
            heading_score=p.heading_score,
            historical_similarity_score=p.historical_similarity_score,
            ai_confidence_adjustment=p.ai_confidence_adjustment,
            ai_reasoning=p.ai_reasoning,
            status=p.status,
            prediction_time=p.prediction_time
        ) for p in predictions
    ]

@router.get("/{prediction_id}", response_model=PredictionDetail)
async def get_prediction(
    prediction_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific prediction details"""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    return PredictionDetail(
        id=prediction.id,
        vessel_id=prediction.vessel_id,
        vessel_name=prediction.vessel.name,
        terminal_id=prediction.terminal_id,
        terminal_name=prediction.terminal.name,
        confidence_score=prediction.confidence_score,
        distance_to_terminal_km=prediction.distance_to_terminal_km,
        eta_hours=prediction.eta_hours,
        predicted_arrival=prediction.predicted_arrival,
        proximity_score=prediction.proximity_score,
        speed_score=prediction.speed_score,
        heading_score=prediction.heading_score,
        historical_similarity_score=prediction.historical_similarity_score,
        ai_confidence_adjustment=prediction.ai_confidence_adjustment,
        ai_reasoning=prediction.ai_reasoning,
        status=prediction.status,
        prediction_time=prediction.prediction_time
    )

@router.post("/{prediction_id}/confirm")
async def confirm_prediction_outcome(
    prediction_id: int,
    request: ConfirmOutcomeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Confirm prediction outcome (triggers learning)"""
    from datetime import datetime

    learning = LearningService(db)
    await learning.process_prediction_outcome(
        prediction_id=prediction_id,
        actual_arrival_time=request.actual_arrival_time or datetime.utcnow(),
        outcome_status=request.status
    )

    return {"message": f"Prediction {prediction_id} outcome processed"}
```

---

## 5.5. WebSocket Manager

### Create `service/src/services/websocket_manager.py`:

```python
from fastapi import WebSocket
from typing import Dict, Set
import json
import logging

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of vessel_ids

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"WebSocket client {client_id} connected")

    async def disconnect(self, client_id: str):
        """Remove connection and subscriptions"""
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        logger.info(f"WebSocket client {client_id} disconnected")

    async def subscribe(self, client_id: str, vessel_ids: list):
        """Subscribe client to specific vessels (empty = all vessels)"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] = set(vessel_ids) if vessel_ids else set()
            logger.debug(f"Client {client_id} subscribed to {len(vessel_ids or [])} vessels")

    async def send_to_client(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast_vessel_update(self, vessel_dict: dict):
        """Broadcast vessel update to subscribed clients"""
        vessel_id = vessel_dict.get('id')
        message = {
            "type": "vessel_update",
            "data": vessel_dict
        }

        for client_id, subscribed_vessels in self.subscriptions.items():
            # Send if subscribed to this vessel or subscribed to all (empty set)
            if not subscribed_vessels or vessel_id in subscribed_vessels:
                await self.send_to_client(client_id, message)

    async def broadcast_prediction(self, prediction_dict: dict):
        """Broadcast new prediction to all clients"""
        message = {
            "type": "prediction_created",
            "data": prediction_dict
        }

        for client_id in self.active_connections.keys():
            await self.send_to_client(client_id, message)


# Global manager instance
manager = WebSocketManager()
```

### Create `service/src/api/routers/websocket.py`:

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uuid
import logging

from src.services.websocket_manager import manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/vessels")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time vessel updates.

    Client messages:
    - {"type": "subscribe", "vessel_ids": ["lng-001", "lng-002"]}
      (empty array = subscribe to all vessels)

    Server messages:
    - {"type": "vessel_update", "data": {...}}
    - {"type": "prediction_created", "data": {...}}
    """
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "subscribe":
                vessel_ids = data.get("vessel_ids", [])
                await manager.subscribe(client_id, vessel_ids)

                # Send confirmation
                await manager.send_to_client(client_id, {
                    "type": "subscribed",
                    "vessel_count": len(vessel_ids) if vessel_ids else "all"
                })

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.disconnect(client_id)
```

---

## 5.6. Admin Router

### Create `service/src/api/routers/admin.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.database.connection import get_db
from src.database.models import Prediction, Vessel, Terminal, VectorEmbedding

router = APIRouter()

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check with database connectivity"""
    try:
        await db.execute(select(func.count()).select_from(Vessel))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": str(e)}

@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """System metrics"""
    # Count vessels
    vessel_count = await db.scalar(select(func.count()).select_from(Vessel))

    # Count terminals
    terminal_count = await db.scalar(select(func.count()).select_from(Terminal))

    # Count predictions by status
    active_predictions = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.status == 'active')
    )
    confirmed_predictions = await db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.status == 'confirmed')
    )

    # Count embeddings
    embedding_count = await db.scalar(select(func.count()).select_from(VectorEmbedding))

    return {
        "vessels": vessel_count,
        "terminals": terminal_count,
        "predictions": {
            "active": active_predictions,
            "confirmed": confirmed_predictions
        },
        "embeddings": embedding_count
    }

@router.get("/accuracy")
async def get_accuracy_stats(db: AsyncSession = Depends(get_db)):
    """Prediction accuracy statistics"""
    # Get confirmed predictions with accuracy scores
    result = await db.execute(
        select(Prediction).where(
            Prediction.status == 'confirmed',
            Prediction.accuracy_score.isnot(None)
        )
    )
    predictions = result.scalars().all()

    if not predictions:
        return {"message": "No confirmed predictions yet"}

    accuracies = [p.accuracy_score for p in predictions]
    avg_accuracy = sum(accuracies) / len(accuracies)

    return {
        "total_confirmed": len(predictions),
        "average_accuracy": avg_accuracy,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies)
    }
```

---

## Verification Checklist

- [ ] FastAPI application starts successfully
- [ ] CORS middleware configured
- [ ] All routers included
- [ ] Swagger docs accessible at `/docs`
- [ ] Vessel CRUD endpoints work
- [ ] Terminal endpoints work
- [ ] `/predictions/analyze` triggers full prediction pipeline
- [ ] `/predictions/active` returns predictions
- [ ] Prediction outcome confirmation works
- [ ] WebSocket connects successfully
- [ ] WebSocket subscription works
- [ ] Broadcast messages reach clients
- [ ] Admin health check responds
- [ ] Admin metrics endpoint works
- [ ] Background tasks start on application startup

---

## Testing

```bash
cd service

# Start server
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Test in browser:
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/health
# - http://localhost:8000/api/v1/vessels
# - http://localhost:8000/api/v1/predictions/active

# Test WebSocket:
# Use wscat or browser WebSocket
```

---

## Next Steps

Once this phase is complete, move to **Phase 6: Frontend Integration** where we'll:
- Create Next.js API client
- Implement WebSocket client
- Update map component to use backend
- Display AI reasoning in UI
- Test full-stack integration

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

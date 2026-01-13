from fastapi import FastAPI
import asyncio
import logging

from src.services.background_tasks import run_all_background_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vessel Track API",
    description="LNG vessel destination prediction system with AI integration",
    version="0.1.0"
)

# Background task handle
background_task = None


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global background_task
    background_task = asyncio.create_task(run_all_background_tasks())
    logger.info("Background tasks started")


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
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Vessel Track API",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "background_tasks": "running" if background_task and not background_task.done() else "stopped"
    }

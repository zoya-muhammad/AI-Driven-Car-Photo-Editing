"""
Car Image AI - Backend
Milestone 1: Upload interface + RMBG-1.4 background removal pipeline
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend directory before any other imports
load_dotenv(Path(__file__).resolve().parent / ".env")

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import LOGS_DIR
from app.routers.process import router as process_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_model():
    """Blocking model load - run in thread pool."""
    from app.services.background_removal import background_removal_service

    background_removal_service._get_pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Pre-load model so requests don't trigger download."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Pre-loading RMBG-1.4 model (this may take 1-2 min on first run)...")
    try:
        await asyncio.to_thread(_load_model)
        logger.info("Car Image AI backend ready")
    except Exception as e:
        logger.error("Failed to load model: %s. Check internet connection and retry.", e)
        raise
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="Car Image AI",
    description="AI-powered car photo editing - background removal with RMBG-1.4",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router)


@app.get("/")
def root():
    return {"message": "Car Image AI API", "docs": "/docs"}

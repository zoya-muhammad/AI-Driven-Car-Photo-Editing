"""
Car Image AI - Backend
Milestone 1: Upload interface + RMBG-1.4 background removal pipeline
"""

from contextlib import asynccontextmanager
import logging

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Car Image AI backend ready")
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

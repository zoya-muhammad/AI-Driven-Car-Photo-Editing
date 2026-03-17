"""Application configuration."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Allowed image types (includes Nikon RAW)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".nef"}
MAX_FILE_SIZE_MB = 35
MAX_BATCH_SIZE = 50

# Gemini API — single model for all car image processing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cleanup: delete logs/outputs older than this (hours) to prevent disk bloat
RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "24"))

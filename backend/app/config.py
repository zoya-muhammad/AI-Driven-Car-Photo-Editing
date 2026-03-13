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

# Hugging Face (optional token for gated models)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
RMBG_MODEL = "briaai/RMBG-1.4"

# Replicate — FLUX.1-Fill-dev (PRIMARY reflection removal model, paid)
# Set REPLICATE_API_TOKEN in backend/.env — required for enhance-preserve mode
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

"""Image processing API routes."""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import ALLOWED_EXTENSIONS, MAX_BATCH_SIZE, MAX_FILE_SIZE_MB
from app.services.processor import (
    get_job_status,
    get_processed_file_path,
    process_sync,
    start_batch,
)

router = APIRouter(prefix="/api", tags=["process"])


def _validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    # Size check happens when reading


@router.post("/process")
async def process_images(files: list[UploadFile] = File(...)):
    """
    Process one or more images for background removal.
    Single image: returns result inline.
    Batch (4+): returns job_id, poll /api/status/{job_id} for progress.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(400, f"Maximum {MAX_BATCH_SIZE} images per batch")

    images: list[tuple[bytes, str]] = []
    for f in files:
        _validate_file(f)
        data = await f.read()
        if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"File {f.filename} exceeds {MAX_FILE_SIZE_MB}MB")
        filename = f.filename or "image.jpg"
        images.append((data, filename))

    # Small batches: sync. Large: async with job_id
    if len(images) <= 3:
        result = process_sync(images)
        return result

    job_id = start_batch(images)
    return {"job_id": job_id, "total": len(images), "message": "Processing started. Poll /api/status/{job_id}"}


@router.get("/status/{job_id}")
async def status(job_id: str):
    """Get processing status and results for a job."""
    data = get_job_status(job_id)
    if not data:
        raise HTTPException(404, "Job not found")
    return data


@router.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    """Download a processed image."""
    path = get_processed_file_path(job_id, filename)
    if not path:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="image/png")

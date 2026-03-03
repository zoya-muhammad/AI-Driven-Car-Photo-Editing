# Car Image AI — Milestone 1

AI-powered car photo editing: upload interface + automatic background removal using RMBG-1.4.

## Milestone 1 Deliverables ✓

### Frontend
- [x] Drag-and-drop image upload
- [x] Single and batch image upload
- [x] Before/after preview slider (with image loading state)
- [x] Download button for processed images
- [x] Simple, clean design for non-technical users
- [x] Progress bar showing processing status
- [x] Loading states (button spinner, skeleton, image placeholders)
- [x] Toast notifications (success/error)
- [x] Responsive layout (mobile-first)

### Backend
- [x] Hugging Face RMBG-1.4 integration
- [x] Basic testing with car photos
- [x] Error handling (failed images flagged)
- [x] Processing logs (`backend/logs/{job_id}.json`)

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0
```

### Frontend

```bash
cd frontend
yarn install
yarn dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_API_URL=http://localhost:8000` if your backend runs elsewhere.

### RMBG-1.4 Model

The model is gated on Hugging Face. Before first run:

1. Go to [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)
2. Accept the license and request access
3. Create a token at [settings/tokens](https://huggingface.co/settings/tokens)
4. Set `HUGGINGFACE_HUB_TOKEN` in your environment or `.env`

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── routers/process.py
│   │   └── services/
│   │       ├── background_removal.py   # RMBG-1.4 pipeline
│   │       └── processor.py             # Job queue, logs
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── DropZone.tsx
│   │   │   ├── BeforeAfterSlider.tsx
│   │   │   ├── ProcessingProgress.tsx
│   │   │   └── ResultGallery.tsx
│   │   └── page.tsx
│   └── package.json
└── README.md
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process` | POST | Upload images (FormData with `files`) |
| `/api/status/{job_id}` | GET | Poll batch job progress |
| `/api/download/{job_id}/{filename}` | GET | Download processed image |

## Processing Logs

Logs are written to `backend/logs/{job_id}.json` for tracking. Failed images are flagged in the API response.

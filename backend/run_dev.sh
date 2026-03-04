#!/bin/bash
# Run backend with reload, excluding uploads/outputs/logs to avoid reload during processing
cd "$(dirname "$0")"
uvicorn main:app --reload --host 0.0.0.0 --port 8000 \
  --reload-exclude "uploads/*" \
  --reload-exclude "outputs/*" \
  --reload-exclude "logs/*"

"""
Pre-download RMBG-1.4 model when network is stable.
Run this before starting the server if you have unreliable internet.
Usage: python download_model.py
"""
import os
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

def main():
    print("Downloading RMBG-1.4 model from Hugging Face...")
    print("This may take 1-2 minutes. Ensure you have a stable internet connection.\n")
    from transformers import pipeline
    pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
    )
    print("\nModel downloaded successfully. You can now start the backend.")

if __name__ == "__main__":
    main()

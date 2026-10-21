FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GOOGLE_CLOUD_LOCATION=us-central1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch wheels first to keep image compact and build fast
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY prosodia /app/prosodia
COPY model /app/model
COPY sample_audio /app/sample_audio
COPY static /app/static
COPY prosodia_lid_engine.py translation_gate.py app.py /app/

# Pre-bake Whisper-small, all-MiniLM-L6-v2, Prosodia checkpoint, and Silero VAD into the image
RUN python3 -c "\
from pathlib import Path; \
from prosodia_lid_engine import ProsodiaLanguageGateEngine; \
from silero_vad import load_silero_vad; \
ProsodiaLanguageGateEngine(ckpt_path=Path('/app/model/best.pt'), bins_path=Path('/app/model/bins.json')); \
load_silero_vad(); \
print('All Prosodia, Whisper, MiniLM, and Silero VAD weights pre-baked and verified!')"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

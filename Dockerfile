# ---------- Stage 1: Base with dependencies ----------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    ffmpeg \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Final image with source ----------
FROM base AS final

WORKDIR /app

# Copy only application source (excluding secrets via .dockerignore)
COPY . .

# Environment variables (secret path is injected at runtime in Cloud Run)
ENV GMAIL_CREDENTIALS_PATH="/secrets/GMAIL_CREDENTIALS"

# Expose port expected by Cloud Run
EXPOSE 8080

# Start the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Real_time_detection:app"]

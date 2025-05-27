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

# ---------- Stage 2: Copy source ----------
FROM base AS final

WORKDIR /app
COPY . .

# Secrets are mounted at runtime
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "Real_time_detection:app"]

# ---------- Stage 1: Base with dependencies ----------
FROM python:3.10-slim AS base

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install essential OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    ffmpeg \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Final image with source ----------
FROM base AS final

# Set working directory again for final image
WORKDIR /app

# Copy application source (excluding secrets via .dockerignore)
COPY . .

# Define runtime environment variable for secret path (mounted by Cloud Run)
ENV GMAIL_CREDENTIALS_PATH="/secrets/GMAIL_CREDENTIALS"

# Expose port for Cloud Run
EXPOSE 8080

# Start the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Real_time_detection:app"]

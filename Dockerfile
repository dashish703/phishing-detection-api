# ---------- Stage 1: Build and install dependencies ----------
FROM python:3.10-slim AS base

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    ffmpeg \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Copy app source and finalize image ----------
FROM base AS final

# Set working directory
WORKDIR /app

# Copy the full application source code
COPY . .

# NOTE: Secrets like GCS_KEY and GMAIL_CREDENTIALS are mounted at runtime on Cloud Run
#       and should NOT be included in this image.

# Expose the port used by Gunicorn on Cloud Run
EXPOSE 8080

# Start the Flask application using Gunicorn
# Replace 'Real_time_detection:app' if your entrypoint module is named differently
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Real_time_detection:app"]

# Stage 1: Build and install dependencies
FROM python:3.10-slim AS base

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy app and credentials
FROM base AS final

# Copy everything including credentials (make sure it's in your .dockerignore exception list if needed)
COPY . .
COPY credentials.json ./credentials.json

# Expose the app port
EXPOSE 8080

# Start app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Real_time_detection:app"]

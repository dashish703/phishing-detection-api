# Use official Python slim image
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy all model and source files
COPY ensemble_phishing_model.pkl .
COPY tfidf_vectorizer.pkl .
COPY xgboost_model.json .
COPY Real_time_detection.py .

# Set environment variable for port (Cloud Run default)
ENV PORT=8080

# Expose port for Cloud Run
EXPOSE 8080

# Set entrypoint
CMD ["python", "Real_time_detection.py"]

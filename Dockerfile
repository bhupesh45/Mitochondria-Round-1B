# Base image targeting AMD64 architecture
FROM --platform=linux/amd64 python:3.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser
WORKDIR /app

# Copy source code and pre-downloaded models
COPY src/ /app/
COPY models/ /app/models/

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Entrypoint to run main.py automatically
CMD ["python", "main.py"]

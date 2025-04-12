FROM python:3.9-slim AS base

# Copy uv binary from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment configs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app

# Copy requirements file first for better caching
COPY requirements.txt .

# First install zenml to use its CLI for getting integration requirements
RUN uv pip install --system -r requirements.txt

# Get integration requirements using zenml CLI and install them
RUN integration_reqs=$(zenml integration export-requirements sklearn gcp numpy) && \
    if [ ! -z "$integration_reqs" ]; then \
        echo "Installing integration requirements: $integration_reqs" && \
        uv pip install --system $integration_reqs "numpy<2.0.0"; \
    fi

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set HOME environment variable
ENV HOME=/home/app

# Expose port
EXPOSE ${PORT}

# Default environment variables for ZenML
ENV SKLEARN_MODEL_NAME="sklearn_model" \
    SKLEARN_MODEL_STAGE="production" \
    PYTORCH_MODEL_NAME="pytorch_model" \
    PYTORCH_MODEL_STAGE="production"

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 

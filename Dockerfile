# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Build arguments
ARG ENVIRONMENT=development

# Set environment variables
ENV ENVIRONMENT=${ENVIRONMENT}

# Copy requirements first for better caching
COPY requirements.txt .

# Install git, install dependencies using secret, then remove git to keep image small
RUN --mount=type=secret,id=gh_token \
    apt-get update && \
    apt-get install -y git && \
    GH_TOKEN=$(cat /run/secrets/gh_token) && \
    sed "s|<GH_TOKEN>|${GH_TOKEN}|g" requirements.txt > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt && \
    rm requirements_temp.txt && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p temp_uploads logs

# Expose port
EXPOSE 80

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Build arguments
ARG GH_TOKEN
ARG ENVIRONMENT=development

# Set environment variables
ENV ENVIRONMENT=${ENVIRONMENT}

# Copy requirements first for better caching
COPY requirements.txt .

# Replace token placeholder with build arg
RUN sed -i "s|<GH_TOKEN>|${GH_TOKEN}|g" requirements.txt

# Install git, install dependencies, then remove git to keep image small
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt && \
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
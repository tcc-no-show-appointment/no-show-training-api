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

# Install git and dependencies, then cleanup
RUN apt-get update && \
    apt-get install -y git libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install basic dependencies and the Microsoft ODBC 17 driver
RUN apt-get update && apt-get install -y gnupg2 curl ca-certificates unixodbc unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p temp_uploads logs

# Expose port
EXPOSE 80

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
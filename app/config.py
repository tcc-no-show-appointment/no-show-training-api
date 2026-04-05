import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    AZURE_STORAGE_ACCOUNT_NAME: str = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "devstoragecenter")
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    
    AZURE_BLOB_CONTAINER_NAME: str = os.getenv("AZURE_BLOB_CONTAINER_NAME", "devconteiner")
    
    API_TITLE: str = "No-Show Training API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for training no-show appointment prediction models"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    UPLOAD_TEMP_DIR: str = os.getenv("UPLOAD_TEMP_DIR", "temp_uploads")
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    # Database Configuration
    # Use connection string if provided, otherwise use individual components
    DB_CONNECTION_STRING: Optional[str] = os.getenv("DB_CONNECTION_STRING")
    DB_SERVER: str = os.getenv("DB_SERVER", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_DRIVER: str = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
    DB_SCHEMA: str = os.getenv("DB_SCHEMA", "dbo")
    DB_TABLE_MODELS_HISTORY: str = os.getenv("DB_TABLE_MODELS_HISTORY", "models_history")
    DB_TABLE_APPOINTMENTS: str = os.getenv("DB_TABLE_APPOINTMENTS", "appointment_predictions")
    DB_TABLE_TRAINING_JOBS: str = os.getenv("DB_TABLE_TRAINING_JOBS", "training_jobs")

    # Parquet-based data on Blob Storage
    BLOB_RAW_APPOINTMENTS_FOLDER: str = os.getenv("BLOB_RAW_APPOINTMENTS_FOLDER", "training-data/raw_appointments")
    BLOB_TRAINING_DATA_FOLDER: str = os.getenv("BLOB_TRAINING_DATA_FOLDER", "training-data/appointment_training_data")
    
    # Training Configuration
    # Limit rows loaded for training (None = load all). Use this for very large datasets (900k+)
    TRAINING_DATA_LIMIT: Optional[int] = int(os.getenv("TRAINING_DATA_LIMIT")) if os.getenv("TRAINING_DATA_LIMIT") else None
    # Limit training data to recent N days (None = all history)
    TRAINING_DAYS_LOOKBACK: Optional[int] = int(os.getenv("TRAINING_DAYS_LOOKBACK")) if os.getenv("TRAINING_DAYS_LOOKBACK") else None

config = Config()

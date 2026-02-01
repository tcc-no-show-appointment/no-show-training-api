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
    DB_SERVER: str = os.getenv("DB_SERVER", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_DRIVER: str = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
    DB_TABLE_MODELS_HISTORY: str = os.getenv("DB_TABLE_MODELS_HISTORY", "tb_models_history")

config = Config()

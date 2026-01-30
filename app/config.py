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

config = Config()

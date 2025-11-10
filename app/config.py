import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    AZURE_STORAGE_ACCOUNT_NAME: str = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "devstoragecenter")
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    
    AZURE_BLOB_CONTAINER_NAME: str = os.getenv("AZURE_BLOB_CONTAINER_NAME", "devconteiner")
    AZURE_BLOB_MODEL_PATH: str = os.getenv("AZURE_BLOB_MODEL_PATH", "trained-models")
    
    API_TITLE: str = "No-Show Training API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for training no-show appointment prediction models"
    
    MODEL_RANDOM_STATE: int = int(os.getenv("MODEL_RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.20"))
    CV_SPLITS: int = int(os.getenv("CV_SPLITS", "3"))
    
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    ALLOWED_EXTENSIONS: set = {".csv", ".xlsx", ".xls", ".parquet"}
    UPLOAD_TEMP_DIR: str = os.getenv("UPLOAD_TEMP_DIR", "temp_uploads")
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

config = Config()

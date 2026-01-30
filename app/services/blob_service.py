from pathlib import Path
from typing import Optional
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError

from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class BlobStorageService:
    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        self.connection_string = connection_string or config.AZURE_STORAGE_CONNECTION_STRING
        self.account_name = account_name or config.AZURE_STORAGE_ACCOUNT_NAME
        self.account_key = account_key or config.AZURE_STORAGE_ACCOUNT_KEY
        self.container_name = container_name or config.AZURE_BLOB_CONTAINER_NAME
        
        self.blob_service_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            if self.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                logger.info("Blob service client initialized with connection string")
            elif self.account_name and self.account_key:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key
                )
                logger.info("Blob service client initialized with account credentials")
            else:
                logger.warning("No Azure credentials provided. Blob upload will not be available.")
        
        except Exception as e:
            logger.error(f"Failed to initialize Blob service client: {str(e)}")
            self.blob_service_client = None
    
    def is_configured(self) -> bool:
        return self.blob_service_client is not None
    
    def upload_model_with_versioning(
        self,
        data: bytes,
        environment: str = "homolog",
        base_name: str = "model"
    ) -> Optional[dict]:
        if not self.is_configured():
            logger.error("Blob Storage is not configured. Cannot upload model.")
            return None
        
        if environment not in ["development","homolog", "prod"]:
            logger.error(f"Invalid environment: {environment}. Must be 'development', 'homolog' or 'prod'.")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_name = f"{environment}/{base_name}_{timestamp}.joblib"
            latest_name = f"{environment}/{base_name}_latest.joblib"
            
            versioned_blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=versioned_name
            )
            
            logger.info(f"Uploading versioned model to: {versioned_name}")
            versioned_blob_client.upload_blob(data, overwrite=False)
            
            latest_blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=latest_name
            )
            
            logger.info(f"Uploading latest copy to: {latest_name}")
            latest_blob_client.upload_blob(
                data, 
                overwrite=True,
                metadata={"original_version": timestamp}
            )
            
            logger.info(f"Model successfully uploaded. Versioned: {versioned_name}, Latest: {latest_name}")
            
            return {
                "versioned_url": versioned_blob_client.url,
                "latest_url": latest_blob_client.url,
                "versioned_name": versioned_name,
                "latest_name": latest_name,
                "timestamp": timestamp,
                "environment": environment
            }
        
        except AzureError as e:
            logger.error(f"Azure error during model upload: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error during model upload: {str(e)}")
            return None
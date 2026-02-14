from pathlib import Path
from typing import Optional
from datetime import datetime
import yaml
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
    
    def download_config_file(
        self,
        folder: str = "model_configuration",
        filename: str = "prod.yaml"
    ) -> Optional[str]:
        """
        Download configuration file from blob storage.
        
        Args:
            folder: Folder name in blob storage (default: 'model_configuration')
            filename: Name of the config file (default: 'prod.yaml')
        
        Returns:
            Configuration file content as string, or None if download fails
        """
        blob_path = f"{folder}/{filename}"
        
        try:
            logger.info(f"Downloading config file from: {blob_path}")
            
            if self.blob_service_client:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_path
                )
                
                download_stream = blob_client.download_blob()
                config_bytes = download_stream.readall()
                config_content = config_bytes.decode('utf-8')
                logger.info(f"Successfully downloaded config file ({len(config_bytes)} bytes)")
                return config_content
            else:
                logger.error("Blob service client not initialized")
                return None
        
        except AzureError as e:
            logger.error(f"Azure error downloading config file: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Error downloading config file from {blob_path}: {str(e)}")
            return None
    
    def get_config_from_blob(
        self,
        folder: str = "model_configuration",
        filename: str = "prod.yaml"
    ) -> dict:
        """
        Download and parse config.yaml from blob storage.
        
        Args:
            folder: Folder name in blob storage (default: 'model_configuration')
            filename: Name of the config file (default: 'prod.yaml')
        
        Returns:
            dict: Parsed configuration dictionary
            
        Raises:
            Exception: If blob storage is not configured or download fails
        """
        logger.info(f"Downloading config from blob storage: {folder}/{filename}")
        
        if not self.is_configured():
            error_msg = "Azure Blob Storage not configured. Cannot download config."
            logger.error(error_msg)
            raise Exception(error_msg)
        
        config_content = self.download_config_file(folder=folder, filename=filename)
        
        if not config_content:
            error_msg = f"Failed to download config file from {folder}/{filename}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        config_dict = yaml.safe_load(config_content)
        logger.info(f"Configuration loaded successfully with keys: {list(config_dict.keys())}")
        return config_dict
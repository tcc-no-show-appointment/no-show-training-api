from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from io import BytesIO
import json
import yaml
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError, ResourceNotFoundError

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
        environment: str = "develop",
        base_name: str = "model"
    ) -> Optional[dict]:
        if not self.is_configured():
            logger.error("Blob Storage is not configured. Cannot upload model.")
            return None
        
        if environment not in ["develop", "homolog", "prod"]:
            logger.error(f"Invalid environment: {environment}. Must be 'develop', 'homolog' or 'prod'.")
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

    def upload_specialty_models(
        self,
        training_output: Dict[str, Dict[str, Any]],
        environment: str = "develop",
        cluster_artifact_bytes: Optional[bytes] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Upload per-specialty model joblibs, consolidated thresholds, and
        optional K-Means cluster artifact to blob storage.

        Path structure:
            {environment}/{specialty_lower}/model_{timestamp}.joblib   (versioned)
            {environment}/{specialty_lower}/model_latest.joblib        (latest)
            {environment}/thresholds/thresholds_{timestamp}.json       (versioned)
            {environment}/thresholds/thresholds_latest.json            (latest)
            {environment}/artifacts/kmeans_cluster_patient_{timestamp}.joblib  (versioned)
            {environment}/artifacts/kmeans_cluster_patient_latest.joblib       (latest)

        Metrics are NOT uploaded to blob storage — they are stored in the DB only.
        Only specialties present in training_output are uploaded; existing
        blobs for other specialties are left untouched.

        Args:
            training_output: Dict from ModelTrainer.train() keyed by specialty.
            environment: Target environment folder.
            cluster_artifact_bytes: Raw bytes of kmeans_cluster_patient.joblib
                produced by noshow_lib v0.4.0 train_model(). Pass None to skip.

        Returns:
            Dict with upload details per specialty + thresholds + artifact, or None on failure.
        """
        if not self.is_configured():
            logger.error("Blob Storage is not configured. Cannot upload models.")
            return None

        if environment not in ["develop", "homolog", "prod"]:
            logger.error(f"Invalid environment: {environment}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_results: Dict[str, Any] = {"specialties": {}, "timestamp": timestamp}

        try:
            for specialty, data in training_output.items():
                specialty_lower = specialty.lower()

                versioned_name = f"{environment}/{specialty_lower}/model_{timestamp}.joblib"
                latest_name = f"{environment}/{specialty_lower}/model_latest.joblib"

                # Upload versioned (never overwrite — unique timestamp)
                versioned_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, blob=versioned_name
                )
                versioned_client.upload_blob(data["model_bytes"], overwrite=False)
                logger.info(f"[{specialty}] Uploaded versioned model: {versioned_name}")

                # Upload latest (always overwrite)
                latest_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, blob=latest_name
                )
                latest_client.upload_blob(
                    data["model_bytes"],
                    overwrite=True,
                    metadata={"original_version": timestamp},
                )
                logger.info(f"[{specialty}] Uploaded latest model: {latest_name}")

                upload_results["specialties"][specialty] = {
                    "versioned_url": versioned_client.url,
                    "latest_url": latest_client.url,
                    "versioned_name": versioned_name,
                    "latest_name": latest_name,
                }

                logger.info(f"[{specialty}] Upload complete")

            # Consolidated thresholds in dedicated folder, versioned
            thresholds = {s: d["threshold"] for s, d in training_output.items()}
            thresholds_bytes = json.dumps(thresholds, indent=2).encode("utf-8")
            thresholds_versioned = f"{environment}/thresholds/thresholds_{timestamp}.json"
            thresholds_latest = f"{environment}/thresholds/thresholds_latest.json"
            self._upload_blob_bytes(thresholds_bytes, thresholds_versioned, overwrite=False)
            self._upload_blob_bytes(thresholds_bytes, thresholds_latest, overwrite=True)
            upload_results["thresholds_versioned"] = thresholds_versioned
            upload_results["thresholds_latest"] = thresholds_latest

            # K-Means cluster artifact (noshow_lib v0.4.0)
            if cluster_artifact_bytes is not None:
                artifact_versioned = (
                    f"{environment}/artifacts/kmeans_cluster_patient_{timestamp}.joblib"
                )
                artifact_latest = (
                    f"{environment}/artifacts/kmeans_cluster_patient_latest.joblib"
                )
                self._upload_blob_bytes(cluster_artifact_bytes, artifact_versioned, overwrite=False)
                self._upload_blob_bytes(cluster_artifact_bytes, artifact_latest, overwrite=True)
                upload_results["cluster_artifact_versioned"] = artifact_versioned
                upload_results["cluster_artifact_latest"] = artifact_latest
                logger.info(f"K-Means cluster artifact uploaded: {artifact_latest}")
            else:
                logger.warning(
                    "No K-Means cluster artifact provided — skipping artifact upload. "
                    "cluster_patient will be -1 at inference time."
                )

            logger.info(
                f"{len(training_output)} specialty model(s) uploaded to '{environment}'"
            )
            return upload_results

        except Exception as e:
            logger.error(f"Error uploading specialty models: {str(e)}", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Internal helper
    # ------------------------------------------------------------------ #

    def _upload_blob_bytes(
        self, data: bytes, blob_name: str, overwrite: bool = True
    ) -> None:
        """Upload raw bytes to a blob."""
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )
        blob_client.upload_blob(data, overwrite=overwrite)

    # ------------------------------------------------------------------ #
    # Parquet raw appointments on Blob Storage
    # ------------------------------------------------------------------ #

    def upload_raw_parquet(
        self,
        df: pd.DataFrame,
        environment: str,
        batch_label: str,
    ) -> str:
        if not self.is_configured():
            raise RuntimeError("Blob Storage is not configured. Cannot upload parquet.")

        blob_name = (
            f"{environment}/{config.BLOB_RAW_APPOINTMENTS_FOLDER}/"
            f"batch_{batch_label}.parquet"
        )

        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
        buffer.seek(0)

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )
        blob_client.upload_blob(buffer, overwrite=False)

        logger.info(
            f"Uploaded raw appointments parquet ({len(df)} rows, "
            f"{buffer.getbuffer().nbytes / 1024:.1f} KB) -> {blob_name}"
        )
        return blob_name

    # ------------------------------------------------------------------ #
    # Parquet training data on Blob Storage
    # ------------------------------------------------------------------ #

    def count_training_blobs(self, environment: str) -> int:
        """Return the number of training Parquet files in Blob Storage for *environment*.

        Does **not** download any data — only lists blob names.  Returns 0 when
        Blob Storage is not configured or when no files are found.
        """
        if not self.is_configured():
            return 0
        prefix = f"{environment}/{config.BLOB_TRAINING_DATA_FOLDER}/"
        container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        return sum(
            1
            for b in container_client.list_blobs(name_starts_with=prefix)
            if b.name.endswith(".parquet")
        )

    def upload_training_parquet(
        self,
        df: pd.DataFrame,
        environment: str,
        batch_label: str,
    ) -> str:
        if not self.is_configured():
            raise RuntimeError("Blob Storage is not configured. Cannot upload parquet.")

        blob_name = (
            f"{environment}/{config.BLOB_TRAINING_DATA_FOLDER}/"
            f"batch_{batch_label}.parquet"
        )

        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
        buffer.seek(0)

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )
        blob_client.upload_blob(buffer, overwrite=False)

        logger.info(
            f"Uploaded training parquet ({len(df)} rows, "
            f"{buffer.getbuffer().nbytes / 1024:.1f} KB) -> {blob_name}"
        )
        return blob_name

    def load_all_training_parquets(
        self,
        environment: str,
        days_lookback: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if not self.is_configured():
            raise RuntimeError("Blob Storage is not configured. Cannot load parquets.")

        prefix = f"{environment}/{config.BLOB_TRAINING_DATA_FOLDER}/"
        container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        blob_names: List[str] = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if blob.name.endswith(".parquet"):
                blob_names.append(blob.name)

        if not blob_names:
            logger.warning(f"No parquet files found under {prefix}")
            return pd.DataFrame()

        blob_names.sort()
        logger.info(f"Found {len(blob_names)} parquet file(s) under {prefix}")

        frames: List[pd.DataFrame] = []
        for name in blob_names:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=name
            )
            stream = blob_client.download_blob()
            buf = BytesIO(stream.readall())
            chunk = pd.read_parquet(buf, engine="pyarrow")
            frames.append(chunk)
            logger.debug(f"Read {len(chunk)} rows from {name}")

        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(df)} total rows from {len(blob_names)} parquet file(s)")

        if days_lookback and "appointment_at" in df.columns:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_lookback)
            df = df[df["appointment_at"] >= cutoff]
            logger.info(
                f"Filtered to {len(df)} rows (last {days_lookback} days)"
            )

        if "appointment_at" in df.columns:
            df = df.sort_values("appointment_at", ascending=False)

        if limit and len(df) > limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} most recent rows")

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Precomputed stats upload
    # ------------------------------------------------------------------ #

    def upload_stats_parquets(
        self,
        patient_stats_df: pd.DataFrame,
        contextual_stats_df: pd.DataFrame,
        environment: str,
    ) -> dict:
        """
        Upload precomputed stats parquets to blob storage.

        Path structure:
            {environment}/stats/patient_stats_{timestamp}.parquet   (versioned)
            {environment}/stats/patient_stats_latest.parquet         (latest)
            {environment}/stats/contextual_stats_{timestamp}.parquet (versioned)
            {environment}/stats/contextual_stats_latest.parquet      (latest)

        Args:
            patient_stats_df: Output of precompute_patient_stats().
            contextual_stats_df: Output of precompute_contextual_stats().
            environment: Target environment folder.

        Returns:
            Dict with upload details.
        """
        if not self.is_configured():
            raise RuntimeError("Blob Storage is not configured. Cannot upload stats.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}

        for name, df in [
            ("patient_stats", patient_stats_df),
            ("contextual_stats", contextual_stats_df),
        ]:
            if df is None or df.empty:
                logger.warning(f"Skipping {name} upload — DataFrame is empty.")
                continue

            versioned_name = f"{environment}/stats/{name}_{timestamp}.parquet"
            latest_name = f"{environment}/stats/{name}_latest.parquet"

            buffer = BytesIO()
            df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
            parquet_bytes = buffer.getvalue()

            self._upload_blob_bytes(parquet_bytes, versioned_name, overwrite=False)
            self._upload_blob_bytes(parquet_bytes, latest_name, overwrite=True)

            results[name] = {
                "versioned_name": versioned_name,
                "latest_name": latest_name,
                "rows": len(df),
                "size_kb": round(len(parquet_bytes) / 1024, 1),
            }
            logger.info(
                f"Uploaded {name}: {len(df)} rows, "
                f"{len(parquet_bytes) / 1024:.1f} KB -> {latest_name}"
            )

        return results
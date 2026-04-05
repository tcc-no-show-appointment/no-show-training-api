from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from app.utils.logger import setup_logger
from app.models.sql_models import ModelRegistry
from app.database import get_db

logger = setup_logger(__name__)


class ModelHistorySaver:
    """Service for saving model training history to database."""
    
    def save_history(
        self,
        db: Session,
        model_name: str,
        metrics: Dict[str, Any],
        blob_url: Optional[str] = None,
        model_version: Optional[str] = None,
        environment: Optional[str] = None,
        file_metadata: Optional[Dict[str, Any]] = None,
        specialty_group: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> ModelRegistry:
        """
        Save model training history to database.
        
        Args:
            db: Database session
            model_name: Name of the model file
            metrics: Dictionary with training metrics
            blob_url: URL to model in blob storage
            model_version: Version identifier
            environment: Environment (development, homolog, prod)
            file_metadata: Metadata about training file
            specialty_group: Specialty group this model was trained for
            threshold: Optimal classification threshold
            
        Returns:
            ModelRegistry: Saved model record
        """
        logger.info(f"Saving model history for: {model_name} (specialty: {specialty_group})")
        
        try:
            model_record = ModelRegistry(
                model_name=model_name,
                model_version=model_version,
                blob_url=blob_url,
                environment=environment or "development",
                specialty_group=specialty_group,
                
                # Metrics
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1_score") or metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                pr_auc=metrics.get("pr_auc"),
                threshold=threshold,
                
                # Store all metrics as JSON for flexibility
                metrics_json=metrics,
                
                # Training info
                training_time_seconds=metrics.get("training_time_seconds"),
                
                # File metadata
                dataset_rows=file_metadata.get("rows") if file_metadata else None,
                dataset_columns=file_metadata.get("columns") if file_metadata else None,
                file_name=file_metadata.get("filename") if file_metadata else None,
                file_size_mb=file_metadata.get("file_size_mb") if file_metadata else None
            )
            
            db.add(model_record)
            db.commit()
            db.refresh(model_record)
            
            logger.info(f"Model history saved successfully. ID: {model_record.id}")
            return model_record
            
        except Exception as e:
            logger.error(f"Failed to save model history: {str(e)}", exc_info=True)
            db.rollback()
            raise

    def save_specialty_histories(
        self,
        db: Session,
        training_output: Dict[str, Dict[str, Any]],
        upload_results: Optional[Dict[str, Any]],
        environment: str,
        file_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ModelRegistry]:
        """
        Save one model history record per trained specialty.

        Args:
            db: Database session
            training_output: Dict from ModelTrainer.train()
            upload_results: Dict from BlobStorageService.upload_specialty_models()
            environment: Environment name
            file_metadata: Metadata about training file

        Returns:
            List of saved ModelRegistry records
        """
        records = []
        timestamp = upload_results.get("timestamp", "") if upload_results else ""

        for specialty, data in training_output.items():
            specialty_lower = specialty.lower()
            model_name = f"lgbm__{specialty_lower}_latest.joblib"

            blob_url = None
            model_version = None
            if upload_results and specialty in upload_results.get("specialties", {}):
                sp_result = upload_results["specialties"][specialty]
                blob_url = sp_result.get("versioned_url")
                model_version = sp_result.get("versioned_name")

            record = self.save_history(
                db=db,
                model_name=model_name,
                metrics=data["metrics"],
                blob_url=blob_url,
                model_version=model_version,
                environment=environment,
                file_metadata=file_metadata,
                specialty_group=specialty,
                threshold=data["threshold"],
            )
            records.append(record)

        logger.info(f"Saved {len(records)} specialty model history records")
        return records

from typing import Dict, Any, Optional
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
        file_metadata: Optional[Dict[str, Any]] = None
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
            file_metadata: Metadata about training file (filename, rows, columns, size)
            
        Returns:
            ModelRegistry: Saved model record
        """
        logger.info(f"Saving model history for: {model_name}")
        
        try:
            # Extract common metrics
            model_record = ModelRegistry(
                model_name=model_name,
                model_version=model_version,
                blob_url=blob_url,
                environment=environment or "development",
                
                # Metrics
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1_score"),
                roc_auc=metrics.get("roc_auc"),
                
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

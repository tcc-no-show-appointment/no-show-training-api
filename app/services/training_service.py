import pandas as pd
import joblib
from io import BytesIO
from typing import Dict, Any
from app.utils.logger import setup_logger
from noshow_lib import train_model

logger = setup_logger(__name__)


class ModelTrainer:
    """Service for model training using noshow_lib."""
    
    def train(
        self, 
        features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info("Starting model training with noshow_lib")
        
        try:
            # train_model returns a dict with artifacts (model object, metrics, paths)
            training_artifacts = train_model(
                df=features,
                config=config
            )
            
            # Extract model object and serialize to bytes
            model = training_artifacts.get("model")
            metrics = training_artifacts.get("metrics", {})
            
            if model is None:
                raise ValueError("Training completed but no model object was returned")
            
            # Serialize model to bytes
            buffer = BytesIO()
            joblib.dump(model, buffer)
            model_bytes = buffer.getvalue()
            
            logger.info(f"Model training completed successfully. Model size: {len(model_bytes)} bytes")
            
            return {
                "model_bytes": model_bytes,
                "metrics": metrics,
                "artifacts": training_artifacts  # Include full artifacts for additional info
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model training failed: {str(e)}") from e


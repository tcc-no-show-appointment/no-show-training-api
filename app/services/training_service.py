import pandas as pd
from typing import Dict, Any
from app.utils.logger import setup_logger
from noshow_lib import train_model

logger = setup_logger(__name__)


class ModelTrainer:
    """Service for model training using noshow_lib."""
    
    def train(
        self, 
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        logger.info("Starting model training with noshow_lib")
        
        try:
            model_bytes, metrics = train_model(
                df_input=features,
                is_external_access=True
            )
            
            logger.info(f"Model training completed successfully. Model size: {len(model_bytes)} bytes")
            
            return {
                "model_bytes": model_bytes,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model training failed: {str(e)}") from e


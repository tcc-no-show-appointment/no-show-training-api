import pickle
from pathlib import Path
from typing import Any, Optional
from app.utils.logger import setup_logger
from app.utils.helpers import generate_model_filename, ensure_directory_exists

logger = setup_logger(__name__)


class ModelSaver:
    def __init__(self, output_dir: str = "models"):
        self.output_dir = output_dir
        ensure_directory_exists(output_dir)
    
    def save_model(
        self,
        model: Any,
        filename: Optional[str] = None,
        base_name: str = "rf_grid_model"
    ) -> str:
        if filename is None:
            filename = generate_model_filename(base_name)
        
        if not filename.endswith('.pkl'):
            filename = f"{filename}.pkl"
        
        filepath = Path(self.output_dir) / filename
        
        logger.info(f"Saving model to {filepath}")
        
        try:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            
            logger.info(f"Model successfully saved to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model save failed: {str(e)}")

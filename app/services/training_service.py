import pandas as pd
import joblib
import json
import shutil
import tempfile
from io import BytesIO
from typing import Dict, Any
from app.utils.logger import setup_logger
from noshow_lib import train_model

logger = setup_logger(__name__)


class ModelTrainer:
    """Service for model training using noshow_lib (per-specialty models)."""
    
    def train(
        self, 
        features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train one LightGBM model per specialty_group using noshow_lib.

        noshow_lib always writes joblib/json artifacts to disk (artifact_dir in
        config). We redirect that to a temp directory and clean it up after
        serializing everything to bytes in memory.

        Returns:
            Dict keyed by specialty_group. Each value contains:
            - model_bytes: serialized joblib bytes
            - metrics_bytes: JSON-encoded metrics + threshold
            - metrics: dict of metric values
            - threshold: optimal threshold for this specialty
            - artifacts: full result dict from noshow_lib
        """
        logger.info("Starting model training with noshow_lib (per-specialty)")

        tmp_dir = tempfile.mkdtemp(prefix="noshow_train_")
        # Override artifact_dir so noshow_lib writes to our temp folder
        config = {**config}
        config["model_specialty"] = {**config.get("model_specialty", {}), "artifact_dir": tmp_dir}

        try:
            results = train_model(df=features, config=config)
            
            training_output: Dict[str, Dict[str, Any]] = {}
            
            for specialty, result in results.items():
                model = result.get("model")
                metrics = result.get("metrics", {})
                threshold = result.get("threshold", 0.5)
                
                if model is None:
                    logger.warning(f"[{specialty}] Training returned no model, skipping")
                    continue
                
                # Serialize model to bytes
                buffer = BytesIO()
                joblib.dump(model, buffer)
                model_bytes = buffer.getvalue()
                
                # Serialize metrics + threshold to JSON bytes
                metrics_data = {
                    "metrics": metrics,
                    "threshold": threshold,
                    "best_params": result.get("best_params", {}),
                    "features_used": result.get("features_used", []),
                    "cat_features": result.get("cat_features", []),
                }
                metrics_bytes = json.dumps(
                    metrics_data, indent=2, ensure_ascii=False
                ).encode("utf-8")
                
                training_output[specialty] = {
                    "model_bytes": model_bytes,
                    "metrics_bytes": metrics_bytes,
                    "metrics": metrics,
                    "threshold": threshold,
                    "artifacts": result,
                }
                
                logger.info(
                    f"[{specialty}] Model size: {len(model_bytes)} bytes | "
                    f"PR-AUC: {metrics.get('pr_auc', 'N/A')} | Threshold: {threshold:.2f}"
                )
            
            if not training_output:
                raise ValueError("Training completed but no models were produced")
            
            logger.info(
                f"Training completed for {len(training_output)} specialties: "
                f"{list(training_output.keys())}"
            )
            return training_output
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model training failed: {str(e)}") from e

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp artifact dir: {tmp_dir}")


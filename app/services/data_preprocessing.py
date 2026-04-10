import pandas as pd
from typing import Dict, Any, Optional
from app.utils.logger import setup_logger
from noshow_lib.feature_engineering import build_features

logger = setup_logger(__name__)


def engineer_features(
    processed_data: pd.DataFrame,
    config: Dict[str, Any],
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Engineer features using noshow_lib's build_features.
    This replaces the old process of load_and_process_data + build_features separately.

    Args:
        processed_data: Raw input dataframe
        config: Configuration dictionary from config.yaml
        history_df: Optional DataFrame with historical appointments (including Status/outcome).
                    Used to compute patient history features correctly during inference.

    Returns:
        DataFrame with engineered features ready for training
    """
    logger.info("Engineering features using noshow_lib.build_features")

    features = build_features(processed_data, config, history_df=history_df)
    logger.info(f"Feature engineering complete. Shape: {features.shape}")

    return features


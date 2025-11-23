import pandas as pd
from typing import Dict, Any
from app.utils.logger import setup_logger
from noshow_lib import load_and_process_data, build_features

logger = setup_logger(__name__)


def process_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw data using noshow_lib."""
    logger.info("Processing raw data using noshow_lib")
    
    if 'AppointmentID' not in df.columns:
        df['AppointmentID'] = range(len(df))
    if 'No-show' not in df.columns:
        df['No-show'] = 'No'
    
    processed_data = load_and_process_data(df)
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    return processed_data


def engineer_features(processed_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features using noshow_lib."""
    logger.info("Engineering features using noshow_lib")
    
    features = build_features(processed_data, {"target_column": "No-show"})
    logger.info(f"Feature engineering complete. Shape: {features.shape}")
    
    return features

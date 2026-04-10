import pandas as pd
from typing import Dict, Any, Optional
from app.utils.logger import setup_logger
from noshow_lib.feature_engineering import build_features
from noshow_lib.feature_engineering_duckdb import build_features_from_parquet

logger = setup_logger(__name__)


def engineer_features(
    processed_data: pd.DataFrame,
    config: Dict[str, Any],
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Engineer features using noshow_lib's build_features (pandas).
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


def engineer_features_duckdb(
    input_parquet: str,
    config: Dict[str, Any],
    output_parquet: str,
    history_parquet: Optional[str] = None,
) -> pd.DataFrame:
    """
    Engineer features using noshow_lib's DuckDB pipeline (Parquet in → Parquet out).

    More memory-efficient for large datasets as DuckDB processes data via
    columnar SQL without loading the full dataset into memory.

    Args:
        input_parquet: Path to raw input Parquet file.
        config: Configuration dictionary from config.yaml.
        output_parquet: Path where the features Parquet will be written.
        history_parquet: Optional path to Parquet with historical appointments.

    Returns:
        DataFrame with engineered features (read from output Parquet).
    """
    logger.info(f"Engineering features using DuckDB pipeline: {input_parquet}")

    build_features_from_parquet(
        parquet_path=input_parquet,
        config=config,
        output_path=output_parquet,
        history_path=history_parquet,
    )

    features = pd.read_parquet(output_parquet, engine="pyarrow")
    logger.info(f"DuckDB feature engineering complete. Shape: {features.shape}")

    return features


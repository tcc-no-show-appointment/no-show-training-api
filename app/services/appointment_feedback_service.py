"""
Ephemeral enrichment of training data from dbo.appointment_predictions.

Records stored by the Prediction API (appointments with a confirmed outcome
Realizado/Falta) are queried at training time, passed through feature
engineering in-memory, and concatenated with the Parquet-based historical
dataset sent to the model trainer.

These rows are NEVER uploaded to Blob Storage — that would cause them to
accumulate as duplicates across training runs since the same rows would be
re-queried on every subsequent training.
"""

import pandas as pd
from typing import Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import config
from app.services.data_preprocessing import engineer_features
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Only statuses that represent a definitive outcome can be used for training.
_TRAINABLE_STATUSES = ("Realizado", "Falta")

# appointment_predictions column names → noshow_lib expected input column names.
_COLUMN_RENAMES = {
    "unit_zipcode": "unit_cep",
}


def load_feedback_as_features(db: Session, config_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Query dbo.appointment_predictions for records with a confirmed outcome,
    run feature engineering in-memory, and return the resulting feature DataFrame.

    Args:
        db:          Active SQLAlchemy session (same DB connection used by the training API).
        config_dict: noshow_lib configuration dict (from blob storage).

    Returns:
        DataFrame with engineered features ready to be concatenated with the
        Parquet training dataset. Returns an empty DataFrame if there are no
        qualifying records or if any step fails.
    """
    table = f"{config.DB_SCHEMA}.{config.DB_TABLE_APPOINTMENTS}"

    stmt = text(f"""  # nosec B608 — table/schema names come from env vars, not user input; values are parameterized
        SELECT
            appointment_prediction_id  AS appointment_id,
            patient_id,
            appointment_status,
            scheduled_at,
            appointment_at,
            patient_age,
            patient_sex,
            patient_city,
            patient_neighborhood,
            insurance_type,
            unit_name,
            unit_address,
            unit_zipcode,
            specialty
        FROM {table}
        WHERE appointment_status IN (:status1, :status2)
          AND scheduled_at   IS NOT NULL
          AND appointment_at IS NOT NULL
    """)

    try:
        engine = db.get_bind()
        df = pd.read_sql(
            stmt,
            engine,
            params={"status1": _TRAINABLE_STATUSES[0], "status2": _TRAINABLE_STATUSES[1]},
        )
    except Exception as e:
        logger.error(f"Failed to query {table}: {e}", exc_info=True)
        return pd.DataFrame()

    if df.empty:
        logger.info("No feedback records in appointment_predictions — skipping DB enrichment")
        return df

    logger.info(f"Loaded {len(df)} feedback records from {table}")

    required_for_fe = ["patient_id", "scheduled_at", "appointment_at"]
    before = len(df)
    df = df.dropna(subset=required_for_fe)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            f"Dropped {dropped} feedback record(s) with NULL in {required_for_fe}"
        )

    if df.empty:
        logger.info("No valid feedback records after sanitisation — skipping DB enrichment")
        return df

    # Derive the target column from the appointment status.
    # Falta (no-show) → 1, Realizado (showed up) → 0
    df["no_show"] = (df["appointment_status"] == "Falta").astype(int)

    # Align column names with what noshow_lib's build_features expects.
    df = df.rename(columns=_COLUMN_RENAMES)

    try:
        features = engineer_features(df, config_dict)
    except Exception as e:
        logger.error(
            f"Feature engineering failed on feedback records: {e}", exc_info=True
        )
        return pd.DataFrame()

    logger.info(
        f"Feedback feature engineering complete — {len(features)} rows ready"
    )
    return features

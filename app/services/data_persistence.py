"""
Service for persisting training pipeline data to the database.

Handles two responsibilities:
1. Save raw CSV data to raw_appointments (staging).
2. Save engineered features to appointment_training_data (feature store).

Uses SQL Server's OUTPUT clause to capture auto-generated IDs without
additional SELECT queries, maintaining traceability via row indexes.
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.config import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

# Raw CSV column → raw_appointments DB column
RAW_CSV_TO_DB_MAP = {
    "id": "raw_id_legacy",
    "idUnicoPaciente": "patient_id",
    "Status": "appointment_status",
    "Marcacao": "scheduled_at",
    "DataHoraConsulta": "appointment_at",
    "Idade": "patient_age",
    "Sexo": "patient_sex",
    "CidadePaciente": "patient_city",
    "BairroPaciente": "patient_neighborhood",
    "TipoConvenio": "insurance_type",
    "UnidadeAtendimento": "unit_name",
    "EnderecoUnidadeAtendimento": "unit_address",
    "CEPUnidadeAtendimento": "unit_zipcode",
    "Especialidade": "specialty",
}

# Feature-engineered DataFrame column → appointment_training_data DB column
# All names match 1:1 except unit_cep → unit_zipcode
FEATURES_TO_TRAINING_MAP = {
    # Context columns
    "patient_id": "patient_id",
    "appointment_status": "appointment_status",
    "scheduled_at": "scheduled_at",
    "appointment_at": "appointment_at",
    "patient_age": "patient_age",
    "patient_sex": "patient_sex",
    "patient_city": "patient_city",
    "patient_neighborhood": "patient_neighborhood",
    "insurance_type": "insurance_type",
    "unit_name": "unit_name",
    "unit_address": "unit_address",
    "unit_cep": "unit_zipcode",
    "specialty": "specialty",
    # Target
    "no_show": "no_show",
    # Engineered features
    "waiting_days": "waiting_days",
    "is_same_day": "is_same_day",
    "age_group": "age_group",
    "appointment_weekday": "appointment_weekday",
    "appointment_day_of_month": "appointment_day_of_month",
    "appointment_week_of_month": "appointment_week_of_month",
    "hour_appointment": "hour_appointment",
    "time_of_day": "time_of_day",
    "is_weekend": "is_weekend",
    "is_holiday": "is_holiday",
    "month_sin": "month_sin",
    "month_cos": "month_cos",
    "weekday_sin": "weekday_sin",
    "weekday_cos": "weekday_cos",
    "hour_sin": "hour_sin",
    "hour_cos": "hour_cos",
    "previous_appointments_count": "previous_appointments_count",
    "days_since_last_visit": "days_since_last_visit",
    "appointments_in_same_schedule": "appointments_in_same_schedule",
    "no_show_rate_patient": "no_show_rate_patient",
    "specialty_no_show_rate": "specialty_no_show_rate",
    "unit_no_show_rate": "unit_no_show_rate",
}


class DataPersistenceService:
    """
    Persists pipeline data to the database without interrupting the
    in-memory flow.  Every public method returns the *original* DataFrame
    so callers can keep chaining.
    """

    def __init__(self, db: Session):
        self.db = db
        self._schema = config.DB_SCHEMA

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_raw_appointments(self, raw_df: pd.DataFrame) -> List[int]:
        """
        Persist raw CSV rows into raw_appointments table and capture
        the auto-generated raw_appointment_id values.

        Args:
            raw_df: The DataFrame loaded directly from the uploaded file,
                    with the original CSV column names.

        Returns:
            List of raw_appointment_id values in the same order as the input
            DataFrame rows. Caller should add these to the DataFrame as a
            raw_appointment_id column before feature engineering.
        """
        table_name = config.DB_TABLE_RAW_APPOINTMENTS
        full_table = f"{self._schema}.{table_name}"

        logger.info(f"Persisting {len(raw_df)} rows to {full_table}")

        try:
            db_df = self._map_columns(raw_df, RAW_CSV_TO_DB_MAP)
            
            # Add created_at timestamp (required NOT NULL column in database)
            db_df["created_at"] = datetime.now()
            
            captured_ids = self._bulk_insert_with_output(
                db_df, table_name, "raw_appointment_id"
            )
            
            logger.info(f"Successfully saved {len(db_df)} rows to {full_table}")
            logger.info(f"Captured {len(captured_ids)} raw_appointment_id values: {captured_ids[:5]}...")
            return captured_ids
            
        except Exception as e:
            logger.error(
                f"Failed to save raw appointments to {full_table}: {e}",
                exc_info=True,
            )
            raise

    def save_training_data(
        self,
        features_df: pd.DataFrame,
        source: str = "raw",
    ) -> pd.DataFrame:
        """
        Persist engineered features into appointment_training_data table.

        Args:
            features_df: DataFrame produced by ``engineer_features()``.
                        If raw_appointment_id column is present, it will be
                        used to populate the FK to raw_appointments table.
            source: Traceability hint — ``"raw"`` for CSV-sourced data,
                    ``"prediction"`` for data coming from labeled predictions.

        Returns:
            The same ``features_df`` — unchanged — so the pipeline can
            continue from memory.
        """
        table_name = config.DB_TABLE_TRAINING_DATA
        full_table = f"{self._schema}.{table_name}"

        logger.info(
            f"Persisting {len(features_df)} engineered rows to {full_table} "
            f"(source={source})"
        )

        try:
            db_df = self._map_columns(features_df, FEATURES_TO_TRAINING_MAP)
            
            # Use raw_appointment_id directly if it was preserved through feature engineering
            if "raw_appointment_id" in features_df.columns:
                db_df["raw_appointment_id"] = features_df["raw_appointment_id"]
                valid_count = db_df["raw_appointment_id"].notna().sum()
                logger.info(f"Using raw_appointment_id from DataFrame ({valid_count}/{len(db_df)} rows with valid IDs)")
            else:
                db_df["raw_appointment_id"] = None
                logger.warning("raw_appointment_id column not found in features - FK will be NULL")
            
            # Add created_at timestamp (required NOT NULL column in database)
            db_df["created_at"] = datetime.now()
            
            self._bulk_insert(db_df, table_name)
            logger.info(f"Successfully saved {len(db_df)} rows to {full_table}")
            
        except Exception as e:
            logger.error(
                f"Failed to save training data to {full_table}: {e}",
                exc_info=True,
            )
            raise

        return features_df

    def load_all_training_data(
        self,
        limit: Optional[int] = None,
        days_lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load all training data from appointment_training_data table for model training.

        Args:
            limit: Optional maximum number of rows to fetch (most recent first).
                   Use this to limit memory usage for very large tables.
            days_lookback: Optional number of days to look back from today.
                          E.g., 730 for last 2 years of data.

        Returns:
            DataFrame with all training features, columns mapped back to
            feature names expected by the model (not DB column names).

        Note:
            For large datasets (900k+ rows), this uses chunked reading to
            avoid memory overflow. Data is sorted by appointment_at DESC
            to get most recent data first if using limit.
        """
        table_name = config.DB_TABLE_TRAINING_DATA
        full_table = f"{self._schema}.{table_name}"

        logger.info(f"Loading training data from {full_table}")
        
        # Build query with proper SQL Server syntax
        select_clause = "SELECT"
        if limit:
            select_clause += f" TOP {limit}"
            logger.info(f"Limiting to {limit} most recent rows")
        
        query = f"{select_clause} * FROM {full_table}"
        
        # Add date filter if specified
        if days_lookback:
            cutoff_date = datetime.now() - timedelta(days=days_lookback)
            query += f" WHERE appointment_at >= '{cutoff_date.strftime('%Y-%m-%d')}'"
            logger.info(f"Filtering data from last {days_lookback} days (since {cutoff_date.date()})")
        
        # Add ordering (most recent first)
        query += " ORDER BY appointment_at DESC"
        
        try:
            # Use pandas read_sql with chunking for memory efficiency
            engine = self.db.get_bind()
            
            # For very large datasets, read in chunks and concatenate
            if limit is None or limit > 100000:
                logger.info("Large dataset detected - using chunked reading")
                chunks = []
                chunksize = 50000  # Read 50k rows at a time
                
                for chunk in pd.read_sql(query, engine, chunksize=chunksize):
                    chunks.append(chunk)
                    logger.debug(f"Loaded chunk: {len(chunk)} rows")
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql(query, engine)
            
            logger.info(f"Loaded {len(df)} rows from {full_table}")
            
            # Map DB columns back to feature names (reverse mapping)
            reverse_map = {v: k for k, v in FEATURES_TO_TRAINING_MAP.items()}
            
            # Only rename columns that exist in the DataFrame
            columns_to_rename = {db_col: feat_col for db_col, feat_col in reverse_map.items() if db_col in df.columns}
            df = df.rename(columns=columns_to_rename)
            
            logger.info(f"Mapped {len(columns_to_rename)} columns back to feature names")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load training data from {full_table}: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_columns(
        df: pd.DataFrame,
        column_map: dict,
    ) -> pd.DataFrame:
        """
        Select and rename columns from *df* according to *column_map*.

        Only columns present in both the DataFrame **and** the mapping are
        kept — extra CSV columns are silently dropped.
        """
        available = {
            src: dest
            for src, dest in column_map.items()
            if src in df.columns
        }

        if not available:
            raise ValueError(
                "None of the expected columns were found in the DataFrame. "
                f"Expected any of: {list(column_map.keys())}, "
                f"got: {list(df.columns)}"
            )

        missing = set(column_map.keys()) - set(available.keys())
        if missing:
            logger.warning(
                f"Columns missing from DataFrame (will be NULL): {missing}"
            )

        return df[list(available.keys())].rename(columns=available).copy()

    def _bulk_insert_with_output(
        self,
        df: pd.DataFrame,
        table_name: str,
        id_column: str,
    ) -> List[int]:
        """
        Bulk-insert rows and capture auto-generated identity values using
        SQL Server's OUTPUT clause.

        Args:
            df: DataFrame with columns matching the target table.
            table_name: Target table name (without schema).
            id_column: Name of the IDENTITY column to capture.

        Returns:
            List of generated ID values in the same order as df rows.
        """
        _ODBC_MAX_PARAMS = 2100
        num_cols = len(df.columns)
        safe_chunksize = max(1, (_ODBC_MAX_PARAMS - 1) // num_cols)

        logger.debug(
            f"Bulk insert with OUTPUT into {self._schema}.{table_name}: "
            f"{len(df)} rows, {num_cols} cols, chunksize={safe_chunksize}"
        )

        all_ids = []
        connection = self.db.connection()
        # Use raw DBAPI cursor for ? placeholder support
        raw_connection = connection.connection
        cursor = raw_connection.cursor()

        try:
            for chunk_start in range(0, len(df), safe_chunksize):
                chunk = df.iloc[chunk_start : chunk_start + safe_chunksize]
                
                # Build INSERT ... OUTPUT statement
                col_names = ", ".join(chunk.columns)
                placeholders = ", ".join(["?"] * len(chunk.columns))
                values_clause = ", ".join([f"({placeholders})" for _ in range(len(chunk))])
                
                # Safe: schema/table/columns are internally controlled, VALUES are parameterized
                sql = (
                    f"INSERT INTO {self._schema}.{table_name} ({col_names}) "  # nosec B608
                    f"OUTPUT INSERTED.{id_column} "
                    f"VALUES {values_clause}"
                )
                
                # Flatten chunk values row by row
                params = []
                for _, row in chunk.iterrows():
                    params.extend(row.tolist())
                
                # Execute and capture IDs using raw cursor
                cursor.execute(sql, params)
                chunk_ids = [row[0] for row in cursor.fetchall()]
                all_ids.extend(chunk_ids)
                
                logger.debug(
                    f"Inserted chunk {chunk_start}:{chunk_start + len(chunk)}, "
                    f"captured {len(chunk_ids)} IDs"
                )
            
            # Commit the transaction
            raw_connection.commit()
            logger.debug(f"Transaction committed for {len(all_ids)} rows")
        finally:
            cursor.close()

        if len(all_ids) != len(df):
            raise RuntimeError(
                f"ID count mismatch: inserted {len(df)} rows "
                f"but captured {len(all_ids)} IDs"
            )

        return all_ids

    def _bulk_insert(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Efficiently bulk-insert *df* using ``pandas.to_sql`` on the
        underlying engine connection, respecting the configured schema.

        ODBC Driver 17 for SQL Server limits each statement to 2,100
        parameters when using multi-row INSERT VALUES.  We dynamically
        calculate a safe chunksize with a small buffer:
        ``columns × chunksize < 2,100``.
        """
        _ODBC_MAX_PARAMS = 2100
        num_cols = len(df.columns)
        # Use 2099 to stay strictly under the 2100 limit
        safe_chunksize = max(1, (_ODBC_MAX_PARAMS - 1) // num_cols)

        logger.debug(
            f"Bulk insert into {self._schema}.{table_name}: "
            f"{len(df)} rows, {num_cols} cols, chunksize={safe_chunksize}"
        )

        engine = self.db.get_bind()
        df.to_sql(
            name=table_name,
            con=engine,
            schema=self._schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=safe_chunksize,
        )

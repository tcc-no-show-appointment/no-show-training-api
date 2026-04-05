"""
Service for persisting raw appointment data to the database.

Handles saving raw CSV data to raw_appointments (staging).
Engineered features (appointment_training_data) are now stored as
Parquet files on Azure Blob Storage — see BlobStorageService.

Uses SQL Server's OUTPUT clause to capture auto-generated IDs without
additional SELECT queries, maintaining traceability via row indexes.
"""

import pandas as pd
from typing import List
from datetime import datetime
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


class DataPersistenceService:
    """
    Persists raw appointment data to the database.
    Training feature data is stored as Parquet on Blob Storage.
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

            # CSV date strings can have 7-digit fractional seconds (e.g. "2019-10-15 15:14:00.0000000")
            # which SQL Server ODBC rejects when passed as raw strings. Parse to datetime first.
            for col in ("scheduled_at", "appointment_at"):
                if col in db_df.columns:
                    db_df[col] = pd.to_datetime(db_df[col], errors="coerce")

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

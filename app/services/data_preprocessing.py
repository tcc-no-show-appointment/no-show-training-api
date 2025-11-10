import numpy as np
import pandas as pd
from typing import Tuple
from app.constants import (
    MIN_AGE, MAX_AGE,
    WAITING_DAYS_BINS, WAITING_DAYS_LABELS
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    
    def __init__(self):
        self.waiting_days_bins = WAITING_DAYS_BINS
        self.waiting_days_labels = WAITING_DAYS_LABELS
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting preprocessing. Initial shape: {df.shape}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Apply preprocessing steps in sequence
        df = self._clean_patient_id(df)
        df = self._encode_target(df)
        df = self._filter_age(df)
        df = self._process_dates(df)
        df = self._calculate_waiting_days(df)
        df = self._create_temporal_features(df)
        df = self._encode_handcap(df)
        df = self._create_waiting_buckets(df)
        df = self._create_weekend_feature(df)
        df = self._create_day_part_features(df)
        df = self._create_cyclic_features(df)
        df = self._create_interaction_features(df)
        
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        return df
    
    def _clean_patient_id(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["PatientId"] = df["PatientId"].astype("int64")
        except Exception:
            df["PatientId"] = df["PatientId"].astype(str).str.replace(r"\\.0$", "", regex=True)
            df["PatientId"] = pd.to_numeric(df["PatientId"], errors="coerce").astype("Int64")
        
        logger.info("PatientId cleaned and converted to integer")
        return df
    
    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if df["No-show"].dtype not in (np.int64, np.int32, np.int8):
            df["No-show"] = df["No-show"].map({"No": 0, "Yes": 1}).astype(int)
            logger.info("Target variable encoded: No=0, Yes=1")
        return df
    
    def _filter_age(self, df: pd.DataFrame) -> pd.DataFrame:
        original_count = len(df)
        df = df[(df["Age"] >= MIN_AGE) & (df["Age"] <= MAX_AGE)].copy()
        removed = original_count - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with invalid age")
        
        return df
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce")
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")
        
        original_count = len(df)
        df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])
        removed = original_count - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with null dates")
        
        logger.info("Date columns processed")
        return df
    
    def _calculate_waiting_days(self, df: pd.DataFrame) -> pd.DataFrame:
        df["waiting_days"] = (
            df["AppointmentDay"].dt.floor("D") - df["ScheduledDay"].dt.floor("D")
        ).dt.days
        
        original_count = len(df)
        df = df[df["waiting_days"] >= 0].copy()
        removed = original_count - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with negative waiting_days")
        
        logger.info("waiting_days calculated")
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["scheduled_hour"] = df["ScheduledDay"].dt.hour
        df["appointment_weekday"] = df["AppointmentDay"].dt.dayofweek
        
        logger.info("Temporal features created: scheduled_hour, appointment_weekday")
        return df
    
    def _encode_handcap(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Handcap"] = (df["Handcap"] > 0).astype(int)
        logger.info("Handcap encoded as binary")
        return df
    
    def _create_waiting_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        df["waiting_days_bucket"] = pd.cut(
            df["waiting_days"],
            bins=self.waiting_days_bins,
            labels=self.waiting_days_labels
        )
        logger.info("Waiting days buckets created")
        return df
    
    def _create_weekend_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_weekend"] = df["AppointmentDay"].dt.dayofweek.isin([5, 6]).astype(int)
        logger.info("Weekend feature created")
        return df
    
    def _create_day_part_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["day_part_morning"] = (
            (df["scheduled_hour"] >= 6) & (df["scheduled_hour"] < 12)
        ).astype(int)
        df["day_part_afternoon"] = (
            (df["scheduled_hour"] >= 12) & (df["scheduled_hour"] < 18)
        ).astype(int)
        df["day_part_evening"] = (
            (df["scheduled_hour"] >= 18) | (df["scheduled_hour"] < 6)
        ).astype(int)
        
        logger.info("Day part features created: morning, afternoon, evening")
        return df
    
    def _create_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["weekday_sin"] = np.sin(2 * np.pi * df["appointment_weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["appointment_weekday"] / 7)
        
        logger.info("Cyclic features created: weekday_sin, weekday_cos")
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["wait_x_sms"] = df["waiting_days"] * df["SMS_received"]
        
        logger.info("Interaction feature created: wait_x_sms")
        return df
    
    def prepare_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        from app.constants import DROP_COLUMNS
        
        X = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
        y = df["No-show"].copy()
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y

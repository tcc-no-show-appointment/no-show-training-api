from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, SmallInteger
from datetime import datetime
from app.database import Base
from app.config import config


class ModelRegistry(Base):
    __tablename__ = config.DB_TABLE_MODELS_HISTORY
    __table_args__ = {"schema": config.DB_SCHEMA}

    id = Column("model_id", Integer, primary_key=True, index=True)
    model_name = Column(String(255), index=True, nullable=False)
    model_version = Column(String(100), nullable=True)
    blob_url = Column(Text, nullable=True)
    environment = Column(String(50), nullable=True)
    
    # Training metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    
    # Additional metrics stored as JSON
    metrics_json = Column(JSON, nullable=True)
    
    # Training information
    training_time_seconds = Column(Float, nullable=True)
    dataset_rows = Column(Integer, nullable=True)
    dataset_columns = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=True)
    file_size_mb = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class RawAppointment(Base):
    """Maps to raw_appointments table - staging table for raw CSV data."""
    __tablename__ = config.DB_TABLE_RAW_APPOINTMENTS
    __table_args__ = {"schema": config.DB_SCHEMA}

    raw_appointment_id = Column(Integer, primary_key=True, index=True)
    raw_id_legacy = Column(Integer, nullable=True)
    patient_id = Column(String(100), nullable=True)
    appointment_status = Column(String(50), nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    appointment_at = Column(DateTime, nullable=True)
    patient_age = Column(Integer, nullable=True)
    patient_sex = Column(String(10), nullable=True)
    patient_city = Column(String(150), nullable=True)
    patient_neighborhood = Column(String(150), nullable=True)
    insurance_type = Column(String(100), nullable=True)
    unit_name = Column(String(150), nullable=True)
    unit_address = Column(String(255), nullable=True)
    unit_zipcode = Column(String(20), nullable=True)
    specialty = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


class AppointmentTrainingData(Base):
    """Maps to appointment_training_data table - consolidated feature store for training."""
    __tablename__ = config.DB_TABLE_TRAINING_DATA
    __table_args__ = {"schema": config.DB_SCHEMA}

    training_id = Column(Integer, primary_key=True, index=True)

    # Traceability
    raw_appointment_id = Column(Integer, nullable=True)
    appointment_prediction_id = Column(Integer, nullable=True)

    # Context columns
    patient_id = Column(String(100), nullable=True)
    appointment_status = Column(String(50), nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    appointment_at = Column(DateTime, nullable=True)
    patient_age = Column(Integer, nullable=True)
    patient_sex = Column(String(10), nullable=True)
    patient_city = Column(String(150), nullable=True)
    patient_neighborhood = Column(String(150), nullable=True)
    insurance_type = Column(String(100), nullable=True)
    unit_name = Column(String(150), nullable=True)
    unit_address = Column(String(255), nullable=True)
    unit_zipcode = Column(String(20), nullable=True)
    specialty = Column(String(150), nullable=True)

    # Target
    no_show = Column(SmallInteger, nullable=False)

    # Engineered features
    waiting_days = Column(Integer, nullable=True)
    is_same_day = Column(SmallInteger, nullable=True)
    age_group = Column(String(50), nullable=True)
    appointment_weekday = Column(String(20), nullable=True)
    appointment_day_of_month = Column(Integer, nullable=True)
    appointment_week_of_month = Column(Integer, nullable=True)
    hour_appointment = Column(Integer, nullable=True)
    time_of_day = Column(String(20), nullable=True)
    is_weekend = Column(SmallInteger, nullable=True)
    is_holiday = Column(SmallInteger, nullable=True)
    month_sin = Column(Float, nullable=True)
    month_cos = Column(Float, nullable=True)
    weekday_sin = Column(Float, nullable=True)
    weekday_cos = Column(Float, nullable=True)
    hour_sin = Column(Float, nullable=True)
    hour_cos = Column(Float, nullable=True)
    previous_appointments_count = Column(Integer, nullable=True)
    days_since_last_visit = Column(Integer, nullable=True)
    appointments_in_same_schedule = Column(Integer, nullable=True)
    no_show_rate_patient = Column(Float, nullable=True)
    specialty_no_show_rate = Column(Float, nullable=True)
    unit_no_show_rate = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False)

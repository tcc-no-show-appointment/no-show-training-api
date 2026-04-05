from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from datetime import datetime
from app.database import Base
from app.config import config
import uuid


class ModelRegistry(Base):
    __tablename__ = config.DB_TABLE_MODELS_HISTORY
    __table_args__ = {"schema": config.DB_SCHEMA}

    id = Column("model_id", Integer, primary_key=True, index=True)
    model_name = Column(String(255), index=True, nullable=False)
    model_version = Column(String(100), nullable=True)
    blob_url = Column(Text, nullable=True)
    environment = Column(String(50), nullable=True)
    specialty_group = Column(String(100), nullable=True)
    
    # Training metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    
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


class TrainingJob(Base):
    __tablename__ = config.DB_TABLE_TRAINING_JOBS
    __table_args__ = {"schema": config.DB_SCHEMA}

    job_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(50), nullable=False, default="pending")  # pending / running / success / failed
    result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    original_filename = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

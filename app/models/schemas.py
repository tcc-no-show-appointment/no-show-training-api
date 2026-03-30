from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingResponse(BaseModel):
    status: str = Field(..., description="Status of the training operation")
    message: str = Field(..., description="Descriptive message")
    specialties_trained: Optional[list] = Field(None, description="List of specialty groups that were trained")
    blob_urls: Optional[Dict[str, Any]] = Field(None, description="Blob URLs per specialty")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics per specialty")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Optimal thresholds per specialty")
    training_time_seconds: Optional[float] = Field(None, description="Time taken to train")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the file is valid")
    file_format: Optional[str] = Field(None, description="Detected file format")
    file_size_mb: Optional[float] = Field(None, description="File size in megabytes")
    rows: Optional[int] = Field(None, description="Number of rows in the dataset")
    columns: Optional[int] = Field(None, description="Number of columns")
    missing_columns: Optional[list] = Field(None, description="List of missing required columns")
    errors: Optional[list] = Field(None, description="List of validation errors")
    warnings: Optional[list] = Field(None, description="List of warnings")


class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class ModelResponse(BaseModel):
    id: int = Field(..., description="Unique identifier of the model")
    model_name: str = Field(..., description="Name of the model")
    model_version: Optional[str] = Field(None, description="Model version identifier")
    blob_url: Optional[str] = Field(None, description="URL to model in blob storage")
    environment: Optional[str] = Field(None, description="Environment (development/homolog/prod)")
    specialty_group: Optional[str] = Field(None, description="Specialty group this model was trained for")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="Model F1 score")
    roc_auc: Optional[float] = Field(None, description="Model ROC AUC score")
    pr_auc: Optional[float] = Field(None, description="Model PR AUC score")
    threshold: Optional[float] = Field(None, description="Optimal classification threshold")
    training_time_seconds: Optional[float] = Field(None, description="Training duration")
    dataset_rows: Optional[int] = Field(None, description="Number of training rows")
    dataset_columns: Optional[int] = Field(None, description="Number of features")
    created_at: datetime = Field(..., description="Creation date of the model")
    
    class Config:
        from_attributes = True
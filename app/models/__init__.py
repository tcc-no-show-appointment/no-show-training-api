"""Models package initialization."""
from app.models.schemas import (
    TrainingResponse,
    TrainingJobAccepted,
    TrainingJobStatus,
    ValidationResponse,
    ErrorResponse,
    HealthResponse
)
from app.models.sql_models import ModelRegistry, TrainingJob  # noqa: F401 — ensures tables are registered with Base

__all__ = [
    "TrainingResponse",
    "ValidationResponse",
    "ErrorResponse",
    "HealthResponse"
]

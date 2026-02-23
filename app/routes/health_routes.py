from pathlib import Path
from fastapi import APIRouter
from app.config import config
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION
    )
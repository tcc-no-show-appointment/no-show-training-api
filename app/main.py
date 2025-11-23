from fastapi import FastAPI

from app.config import config
from app.routes import training_router
from app.models.schemas import HealthResponse
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

app.include_router(training_router)

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION
    )

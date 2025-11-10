from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import config
from app.routes import training_router
from app.models.schemas import HealthResponse, ErrorResponse
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            message="An unexpected error occurred",
            details={"error": str(exc)}
        ).dict()
    )

app.include_router(training_router)

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION
    )

@app.get("/info")
async def api_info():
    return {
        "title": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "allowed_extensions": list(config.ALLOWED_EXTENSIONS),
        "endpoints": {
            "train": "/training/upload-and-train",
            "validate": "/training/validate",
            "health": "/health"
        }
    }

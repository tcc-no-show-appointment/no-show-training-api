from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import config
from app.routes import training_router, health_router, model_history_router
from app.models.schemas import HealthResponse
from app.utils.logger import setup_logger
from app.database import engine, Base

logger = setup_logger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    redirect_slashes=False
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

app.include_router(training_router)
app.include_router(health_router)
app.include_router(model_history_router)

@app.get("/", response_model=HealthResponse)
@app.get("", response_model=HealthResponse, include_in_schema=False)
async def root():
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION
    )

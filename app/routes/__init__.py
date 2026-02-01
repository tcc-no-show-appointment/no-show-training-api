"""Routes package initialization."""
from app.routes.training_routes import router as training_router
from app.routes.health_routes import router as health_router
from app.routes.model_history_routes import router as model_history_router

__all__ = ["training_router", "health_router", "model_history_router"]   

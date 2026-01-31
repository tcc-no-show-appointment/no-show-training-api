"""Routes package initialization."""
from app.routes.training_routes import router as training_router
from app.routes.health_routes import router as health_router

__all__ = ["training_router", "health_router"]
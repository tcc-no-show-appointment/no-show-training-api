from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.schemas import ModelResponse
from app.models.sql_models import ModelRegistry

router = APIRouter(prefix="/model-history", tags=["Model History"])

@router.get("/", response_model=List[ModelResponse])
async def get_all_models(db: Session = Depends(get_db)):
    """
    Get all models registered in the database.
    Returns id, model name, created date and accuracy.
    """
    models = db.query(ModelRegistry).all()
    return models

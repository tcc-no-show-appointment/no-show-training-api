from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.database import Base
from app.config import config

class ModelRegistry(Base):
    __tablename__ = config.DB_TABLE_MODELS_HISTORY

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now)
    accuracy = Column(Float, nullable=True)

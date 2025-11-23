import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.training_service import ModelTrainer


def test_model_trainer_initialization():
    """Test ModelTrainer can be initialized"""
    trainer = ModelTrainer()
    assert trainer is not None

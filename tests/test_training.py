import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.training_service import ModelTrainer


class TestModelTrainer:
    def test_initialization(self):
        """ModelTrainer can be instantiated without arguments."""
        trainer = ModelTrainer()
        assert trainer is not None

    def test_has_train_method(self):
        """ModelTrainer exposes a .train() callable."""
        trainer = ModelTrainer()
        assert callable(getattr(trainer, "train", None))

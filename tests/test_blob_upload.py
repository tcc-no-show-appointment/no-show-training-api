import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.blob_service import BlobStorageService


def test_blob_service_initialization():
    """Test BlobStorageService can be initialized"""
    service = BlobStorageService()
    assert service is not None


def test_blob_service_configuration_check():
    """Test configuration check method"""
    service = BlobStorageService()
    result = service.is_configured()
    assert isinstance(result, bool)

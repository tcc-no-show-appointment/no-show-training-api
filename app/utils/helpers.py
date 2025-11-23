"""Helper functions for file handling and utilities."""
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from app.constants import SUPPORTED_FORMATS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_file_extension(filename: str) -> bool:
    ext = get_file_extension(filename)
    return ext in [f".{fmt}" for fmt in SUPPORTED_FORMATS.keys()]


def generate_unique_filename(original_filename: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    ext = get_file_extension(original_filename)
    base_name = Path(original_filename).stem
    
    return f"{base_name}_{timestamp}_{unique_id}{ext}"


def generate_model_filename(base_name: str = "rf_grid_model", environment: Optional[str] = None) -> str:
    if environment:
        return f"{base_name}_{environment}.joblib"
    return f"{base_name}.joblib"


def ensure_directory_exists(directory: str) -> Path:
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {dir_path}")
    return dir_path


def get_file_size_mb(file_path: str) -> float:
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def cleanup_temp_file(file_path: str) -> bool:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")
        return False


def detect_file_format(filename: str) -> Optional[str]:
    ext = get_file_extension(filename)
    format_map = {
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".parquet": "parquet"
    }
    return format_map.get(ext)

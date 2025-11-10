import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.constants import REQUIRED_COLUMNS, SUPPORTED_FORMATS
from app.utils.logger import setup_logger
from app.utils.helpers import detect_file_format, get_file_size_mb

logger = setup_logger(__name__)


class DataValidator:
    def __init__(self):
        self.required_columns = REQUIRED_COLUMNS
        self.supported_formats = SUPPORTED_FORMATS
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        validation_result = {
            "is_valid": False,
            "file_format": None,
            "file_size_mb": None,
            "rows": None,
            "columns": None,
            "missing_columns": [],
            "errors": [],
            "warnings": []
        }
        
        # Check if file exists
        if not Path(file_path).exists():
            validation_result["errors"].append("File does not exist")
            logger.error(f"Validation failed: File does not exist - {file_path}")
            return validation_result
        
        # Get file size
        try:
            validation_result["file_size_mb"] = get_file_size_mb(file_path)
        except Exception as e:
            validation_result["warnings"].append(f"Could not determine file size: {str(e)}")
        
        # Detect file format
        file_format = detect_file_format(file_path)
        validation_result["file_format"] = file_format
        
        if file_format is None:
            validation_result["errors"].append(
                f"Unsupported file format. Supported formats: {list(self.supported_formats.keys())}"
            )
            logger.error(f"Validation failed: Unsupported file format - {file_path}")
            return validation_result
        
        # Try to load the file
        try:
            df = self._load_file(file_path, file_format)
            logger.info(f"Successfully loaded file. Shape: {df.shape}")
        except Exception as e:
            validation_result["errors"].append(f"Failed to load file: {str(e)}")
            logger.error(f"Validation failed: Could not load file - {str(e)}")
            return validation_result
        
        # Check dataframe
        validation_result["rows"] = len(df)
        validation_result["columns"] = len(df.columns)
        
        # Validate schema
        schema_validation = self._validate_schema(df)
        validation_result["missing_columns"] = schema_validation["missing_columns"]
        validation_result["errors"].extend(schema_validation["errors"])
        validation_result["warnings"].extend(schema_validation["warnings"])
        
        # Additional data quality checks
        quality_checks = self._check_data_quality(df)
        validation_result["warnings"].extend(quality_checks["warnings"])
        
        # Determine if valid
        validation_result["is_valid"] = len(validation_result["errors"]) == 0
        
        if validation_result["is_valid"]:
            logger.info(f"File validation successful: {file_path}")
        else:
            logger.warning(f"File validation failed: {file_path}")
            logger.warning(f"Errors: {validation_result['errors']}")
        
        return validation_result
    
    def _load_file(self, file_path: str, file_format: str) -> pd.DataFrame:
        if file_format == "csv":
            return pd.read_csv(file_path)
        elif file_format in ["xlsx", "xls"]:
            return pd.read_excel(file_path)
        elif file_format == "parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            "missing_columns": [],
            "errors": [],
            "warnings": []
        }
        
        # Check for missing required columns
        missing = [col for col in self.required_columns if col not in df.columns]
        
        if missing:
            result["missing_columns"] = missing
            result["errors"].append(
                f"Missing required columns: {', '.join(missing)}"
            )
            logger.error(f"Schema validation failed: Missing columns {missing}")
        
        # Check for extra columns (warning only)
        extra = [col for col in df.columns if col not in self.required_columns]
        if extra:
            result["warnings"].append(
                f"Extra columns found (will be ignored): {', '.join(extra)}"
            )
        
        return result
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            "warnings": []
        }
        
        # Check for empty dataframe
        if len(df) == 0:
            result["warnings"].append("Dataframe is empty")
            return result
        
        # Check for missing values in critical columns
        critical_columns = ["No-show", "Age", "Gender"]
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    result["warnings"].append(
                        f"Column '{col}' has {missing_count} missing values ({missing_pct:.2f}%)"
                    )
        
        # Check target variable distribution (if present)
        if "No-show" in df.columns:
            try:
                value_counts = df["No-show"].value_counts()
                if len(value_counts) < 2:
                    result["warnings"].append(
                        "Target variable 'No-show' has only one class"
                    )
                else:
                    # Check for severe imbalance
                    min_class_pct = (value_counts.min() / len(df)) * 100
                    if min_class_pct < 1:
                        result["warnings"].append(
                            f"Severe class imbalance detected in 'No-show': minority class is {min_class_pct:.2f}%"
                        )
            except Exception as e:
                result["warnings"].append(f"Could not analyze target variable: {str(e)}")
        
        return result
    
    def load_and_validate(self, file_path: str) -> tuple:
        validation_result = self.validate_file(file_path)
        
        if validation_result["is_valid"]:
            try:
                file_format = validation_result["file_format"]
                df = self._load_file(file_path, file_format)
                return validation_result, df
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Failed to load validated file: {str(e)}")
                return validation_result, None
        
        return validation_result, None

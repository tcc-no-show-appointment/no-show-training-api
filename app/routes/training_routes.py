import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.config import config
from app.models.schemas import TrainingResponse, ValidationResponse
from app.services import (
    DataValidator,
    DataPreprocessor,
    ModelTrainer,
    ModelSaver,
    BlobStorageService
)
from app.utils.logger import setup_logger
from app.utils.helpers import (
    generate_unique_filename,
    ensure_directory_exists,
    cleanup_temp_file,
    validate_file_extension
)

logger = setup_logger(__name__)
router = APIRouter(prefix="/training", tags=["Training"])

@router.post("/upload-and-train", response_model=TrainingResponse)
async def upload_and_train(
    file: UploadFile = File(..., description="Data file (CSV, Excel, or Parquet)")
):
    temp_file_path = None
    model_file_path = None
    
    try:
        # Step 1: Validate file extension
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {list(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Step 2: Save uploaded file temporarily
        temp_dir = ensure_directory_exists(config.UPLOAD_TEMP_DIR)
        unique_filename = generate_unique_filename(file.filename)
        temp_file_path = str(temp_dir / unique_filename)
        
        logger.info(f"Saving uploaded file to {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Step 3: Validate data
        logger.info("Starting data validation")
        validator = DataValidator()
        validation_result, df = validator.load_and_validate(temp_file_path)
        
        if not validation_result["is_valid"]:
            logger.error(f"Data validation failed: {validation_result['errors']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Data validation failed",
                    "errors": validation_result["errors"],
                    "warnings": validation_result.get("warnings", [])
                }
            )
        
        logger.info(f"Data validated successfully. Shape: {df.shape}")
        
        # Step 4: Preprocess data
        logger.info("Starting data preprocessing")
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_data(df)
        X, y = preprocessor.prepare_features_and_target(df_processed)
        
        logger.info(f"Preprocessing complete. Features: {X.shape}, Target: {y.shape}")
        
        # Step 5: Train model with fixed parameters (matching original model.py)
        logger.info("Starting model training with fixed grid search parameters")
        trainer = ModelTrainer()
        
        training_result = trainer.train(
            X, y,
            n_estimators_range=[100, 180],
            max_depth_range=[None, 12],
            min_samples_leaf_range=[10, 20]
        )
        
        logger.info("Model training completed")
        
        # Step 6: Save model with default naming
        logger.info("Saving model to .pkl file")
        model_saver = ModelSaver()
        model_file_path = model_saver.save_model(
            training_result["model"],
            base_name="rf_grid_model"
        )
        
        logger.info(f"Model saved to {model_file_path}")
        
        # Step 7: Upload to Azure Blob Storage
        blob_url = None
        blob_service = BlobStorageService()
        
        if blob_service.is_configured():
            logger.info("Uploading model to Azure Blob Storage as rf_grid_model.pkl")
            blob_url = blob_service.upload_file(
                model_file_path,
                blob_name="rf_grid_model.pkl",
                overwrite=True
            )
            
            if blob_url:
                logger.info(f"Model uploaded to blob: {blob_url}")
            else:
                logger.warning("Failed to upload model to Azure Blob Storage")
        else:
            logger.warning("Azure Blob Storage not configured, skipping upload")
        
        # Prepare response
        response = TrainingResponse(
            status="success",
            message="Model training completed successfully",
            model_filename=Path(model_file_path).name,
            blob_url=blob_url,
            metrics=training_result["metrics"],
            training_time_seconds=training_result["metrics"]["training_time_seconds"]
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Training pipeline failed: {str(e)}"
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)


@router.post("/validate", response_model=ValidationResponse)
async def validate_file_endpoint(
    file: UploadFile = File(..., description="Data file to validate")
):
    temp_file_path = None
    
    try:
        if not validate_file_extension(file.filename):
            return ValidationResponse(
                is_valid=False,
                errors=[f"Unsupported file format. Allowed: {list(config.ALLOWED_EXTENSIONS)}"]
            )
        
        temp_dir = ensure_directory_exists(config.UPLOAD_TEMP_DIR)
        unique_filename = generate_unique_filename(file.filename)
        temp_file_path = str(temp_dir / unique_filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        validator = DataValidator()
        validation_result = validator.validate_file(temp_file_path)
        
        return ValidationResponse(**validation_result)
    
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return ValidationResponse(
            is_valid=False,
            errors=[f"Validation failed: {str(e)}"]
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)

import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import config
from app.models.schemas import TrainingResponse, ValidationResponse
from app.services import (
    DataValidator,
    process_raw_data,
    engineer_features,
    ModelTrainer,
    ModelHistorySaver,
    BlobStorageService
)
from app.utils.logger import setup_logger
from app.utils.helpers import (
    generate_unique_filename,
    ensure_directory_exists,
    cleanup_temp_file
)

logger = setup_logger(__name__)
router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/validate", response_model=ValidationResponse)
async def validate_file_endpoint(
    file: UploadFile = File(..., description="Data file to validate")
):
    """Validate uploaded file without processing."""
    temp_file_path = None
    
    try:
        temp_dir = ensure_directory_exists(config.UPLOAD_TEMP_DIR)
        unique_filename = generate_unique_filename(file.filename)
        temp_file_path = str(temp_dir / unique_filename)
        
        logger.info(f"Saving file for validation: {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        validator = DataValidator()
        validation_result = validator.validate_file(temp_file_path)
        
        logger.info(f"Validation result: {validation_result['is_valid']}")
        
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "File validation failed",
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }
            )
        
        return ValidationResponse(**validation_result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)


@router.post("/upload-and-train", response_model=TrainingResponse)
async def upload_and_train(
    file: UploadFile = File(..., description="Data file (CSV, Excel, or Parquet)")
):
    """Upload file and train model using noshow_lib pipeline."""
    temp_file_path = None
    
    try:
        temp_dir = ensure_directory_exists(config.UPLOAD_TEMP_DIR)
        unique_filename = generate_unique_filename(file.filename)
        temp_file_path = str(temp_dir / unique_filename)
        
        logger.info(f"Saving uploaded file to {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
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
        
        logger.info("Processing raw data using noshow_lib")
        processed_data = process_raw_data(df)
        
        logger.info("Engineering features using noshow_lib")
        features = engineer_features(processed_data)
        
        logger.info("Training model with noshow_lib")
        trainer = ModelTrainer()
        training_result = trainer.train(features=features)
        
        model_bytes = training_result["model_bytes"]
        metrics = training_result["metrics"]
        
        logger.info(f"Training completed. Model size: {len(model_bytes)} bytes, Metrics: {list(metrics.keys())}")
        
        logger.info("Uploading trained model to Azure Blob Storage")
        from app.utils.helpers import generate_model_filename
        
        model_filename = generate_model_filename(
            base_name="noshow_model",
            environment=config.ENVIRONMENT
        )
        
        blob_service = BlobStorageService()
        
        if not blob_service.is_configured():
            logger.warning("Azure Blob Storage not configured - skipping model upload")
            model_url = None
        else:
            model_url = blob_service.upload_bytes(
                data=model_bytes,
                blob_name=model_filename,
                blob_path="models",
                overwrite=True
            )
            
            if model_url:
                logger.info(f"Model uploaded successfully to: {model_url}")
            else:
                logger.warning("Model upload returned no URL")
        
        logger.info("Saving model history")

        # TODO: Save on Database the models history
        # history_saver = ModelHistorySaver()
        # history_saver.save_history( 
        #     model_name=model_filename,
        #     metrics=metrics,
        #     file_metadata={
        #         "filename": file.filename,
        #         "rows": validation_result.get("rows"),
        #         "columns": validation_result.get("columns"),
        #         "file_size_mb": validation_result.get("file_size_mb")
        #     }
        # )
        
        response = TrainingResponse(
            status="success",
            message="Training flow completed successfully",
            model_filename=model_filename,
            blob_url=model_url,
            metrics=metrics,
            training_time_seconds=metrics.get("training_time_seconds", 0)
        )
        
        logger.info("Training flow completed successfully")
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

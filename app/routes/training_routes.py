import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import config
from app.database import get_db
from app.models.schemas import TrainingResponse, ValidationResponse
from app.services import (
    DataValidator,
    engineer_features,
    DataPersistenceService,
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
        
        # Step 1: Download config to get required columns
        blob_service = BlobStorageService()
        try:
            config_dict = blob_service.get_config_from_blob()
            required_columns = config_dict.get("schema", {}).get("required_columns", [])
            logger.info(f"Using {len(required_columns)} required columns from config")
        except Exception as e:
            logger.warning(f"Failed to get config from blob: {str(e)}. Using default validation.")
            required_columns = None
            config_dict = None
        
        # Step 2: Basic validation (file format, structure, required columns)
        validator = DataValidator(required_columns=required_columns)
        validation_result, df = validator.load_and_validate(temp_file_path)
        
        logger.info(f"Basic validation result: {validation_result['is_valid']}")
        
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Basic validation failed",
                    "errors": validation_result.get('errors', []),
                    "warnings": validation_result.get('warnings', [])
                }
            )
        
        # Step 3: Validate with noshow_lib if config was loaded
        if config_dict is not None:
            try:
                validator.validate_with_config(df, config_dict)
                validation_result["warnings"] = validation_result.get("warnings", [])
                validation_result["warnings"].append("Schema validation passed with config requirements")
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Schema validation failed",
                        "errors": [str(e)]
                    }
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Config validation error: {str(e)}"
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
    file: UploadFile = File(..., description="Data file (CSV, Excel, or Parquet)"),
    db: Session = Depends(get_db)
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
        
        # Step 1: Download config to get required columns and for later use
        blob_service = BlobStorageService()
        try:
            config_dict = blob_service.get_config_from_blob()
            required_columns = config_dict.get("schema", {}).get("required_columns", [])
            logger.info(f"Config loaded with {len(required_columns)} required columns")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download configuration: {str(e)}"
            )

        # Step 2: Basic validation (file format, structure, required columns from config)
        logger.info("Starting data validation")
        validator = DataValidator(required_columns=required_columns)
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
        
        # Step 3: Validate with noshow_lib schema
        try:
            validator.validate_with_config(df, config_dict)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Schema validation failed",
                    "errors": [str(e)]
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Config validation error: {str(e)}"
            )
        
        # Step 4: Save raw data to database and capture IDs
        logger.info("Persisting raw data to raw_appointments table")
        persistence = DataPersistenceService(db)
        try:
            captured_ids = persistence.save_raw_appointments(df)
            logger.info(f"Captured {len(captured_ids)} raw_appointment_id values")
            
            # Add the captured IDs to the DataFrame before feature engineering
            df["raw_appointment_id"] = captured_ids
            logger.info(f"Added raw_appointment_id column to DataFrame")
            
            # Commit the session to ensure raw data is persisted
            db.commit()
            logger.info("Raw data transaction committed")
        except Exception as e:
            logger.error(f"Failed to persist raw appointments: {e}", exc_info=True)
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save raw data to database: {str(e)}"
            )

        # Step 5: Feature engineering (will preserve raw_appointment_id column)
        logger.info(f"Before build_features - DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        logger.info(f"Raw appointment IDs present: {df['raw_appointment_id'].head().tolist()}")
        
        features = engineer_features(df, config_dict)
        
        logger.info(f"After build_features - DataFrame shape: {features.shape}, columns: {list(features.columns)}")
        if "raw_appointment_id" in features.columns:
            logger.info(f"[SUCCESS] raw_appointment_id preserved! Sample: {features['raw_appointment_id'].head().tolist()}")
        else:
            logger.warning("[WARNING] raw_appointment_id was NOT preserved by build_features")

        # Step 6: Save training data to database with ID linkage
        logger.info("Persisting training data to appointment_training_data table")
        try:
            persistence.save_training_data(features, source="raw")
            db.commit()
            logger.info("Training data transaction committed")
        except Exception as e:
            logger.error(f"Failed to persist training data: {e}", exc_info=True)
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save training data to database: {str(e)}"
            )

        # Step 7: Validate dataset size for training
        MIN_SAMPLES_TOTAL = 20
        MIN_SAMPLES_PER_CLASS = 2
        target_col = config_dict.get("data", {}).get("target_column", "no_show")
        
        if target_col not in features.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_col}' not found in features after engineering"
            )
        
        class_counts = features[target_col].value_counts()
        total_samples = len(features)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0
        
        logger.info(f"Dataset validation - Total samples: {total_samples}, Class distribution: {class_counts.to_dict()}")
        
        if total_samples < MIN_SAMPLES_TOTAL:
            logger.warning(f"Dataset too small for training: {total_samples} samples (minimum: {MIN_SAMPLES_TOTAL})")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Dataset too small for model training",
                    "total_samples": total_samples,
                    "minimum_required": MIN_SAMPLES_TOTAL,
                    "class_distribution": class_counts.to_dict(),
                    "hint": "Upload a larger dataset with at least 20 rows containing both 'realizado' and 'falta' appointments"
                }
            )
        
        if min_class_count < MIN_SAMPLES_PER_CLASS:
            logger.warning(f"Imbalanced dataset: smallest class has only {min_class_count} sample(s)")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Dataset too imbalanced for stratified training",
                    "class_distribution": class_counts.to_dict(),
                    "minimum_per_class": MIN_SAMPLES_PER_CLASS,
                    "hint": "Ensure each class (realizado/falta) has at least 2 samples"
                }
            )

        # Step 8: Train model
        logger.info("Training model with noshow_lib")
        trainer = ModelTrainer()
        training_result = trainer.train(features=features, config=config_dict)
        
        model_bytes = training_result["model_bytes"]
        metrics = training_result["metrics"]
        
        logger.info(f"Training completed. Model size: {len(model_bytes)} bytes, Metrics: {list(metrics.keys())}")
        
        # Step 9: Upload model to blob storage
        logger.info("Uploading t rained model to Azure Blob Storage with versioning")
        
        if not blob_service.is_configured():
            logger.warning("Azure Blob Storage not configured - skipping model upload")
            model_url = None
            model_filename = None
            timestamp = None
        else:
            result = blob_service.upload_model_with_versioning(
                data=model_bytes,
                environment=config.ENVIRONMENT,
                base_name="model"
            )
            
            if result:
                model_url = result["latest_url"]
                model_filename = result["latest_name"]
                timestamp = result["timestamp"]
                logger.info(f"Model uploaded successfully")
                logger.info(f"  Versioned: {result['versioned_name']}")
                logger.info(f"  Latest: {result['latest_name']}")
            else:
                logger.warning("Model upload failed")
                model_url = None
                model_filename = None
                timestamp = None
        # Step 10: Save model history to database
        logger.info("Saving model history to database")
        try:
            history_saver = ModelHistorySaver()
            model_record = history_saver.save_history(
                db=db,
                model_name=model_filename or "model_unknown",
                metrics=metrics,
                blob_url=model_url,
                model_version=result.get("versioned_name") if result else None,
                environment=config.ENVIRONMENT,
                file_metadata={
                    "filename": file.filename,
                    "rows": validation_result.get("rows"),
                    "columns": validation_result.get("columns"),
                    "file_size_mb": validation_result.get("file_size_mb")
                }
            )
            logger.info(f"Model history saved with ID: {model_record.id}")
        except Exception as e:
            logger.error(f"Failed to save model history: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save model history: {str(e)}"
            )

        
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

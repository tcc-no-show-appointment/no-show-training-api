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

        # Step 7: Load ALL training data (historical + new) for model training
        logger.info("Loading all training data from database for model training")
        try:
            # Use config to control data loading for large datasets
            all_training_data = persistence.load_all_training_data(
                limit=config.TRAINING_DATA_LIMIT,  # None = all data, or set limit (e.g., 500000)
                days_lookback=config.TRAINING_DAYS_LOOKBACK  # None = all history, or days (e.g., 730)
            )
            logger.info(f"Loaded {len(all_training_data)} total rows for training (historical + new)")
            
            if config.TRAINING_DATA_LIMIT:
                logger.info(f"Training data limited to {config.TRAINING_DATA_LIMIT} most recent rows")
            if config.TRAINING_DAYS_LOOKBACK:
                logger.info(f"Training data limited to last {config.TRAINING_DAYS_LOOKBACK} days")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load training data from database: {str(e)}"
            )

        # Step 8: Validate combined dataset size for training
        MIN_SAMPLES_TOTAL = 20
        MIN_SAMPLES_PER_CLASS = 2
        target_col = config_dict.get("data", {}).get("target_column", "no_show")
        
        if target_col not in all_training_data.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_col}' not found in training data"
            )
        
        class_counts = all_training_data[target_col].value_counts()
        total_samples = len(all_training_data)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0
        
        logger.info(f"Combined dataset validation - Total samples: {total_samples}, Class distribution: {class_counts.to_dict()}")
        
        if total_samples < MIN_SAMPLES_TOTAL:
            logger.warning(f"Dataset too small for training: {total_samples} samples (minimum: {MIN_SAMPLES_TOTAL})")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Combined dataset too small for model training",
                    "total_samples": total_samples,
                    "minimum_required": MIN_SAMPLES_TOTAL,
                    "class_distribution": class_counts.to_dict(),
                    "hint": "Need more training data in database"
                }
            )
        
        if min_class_count < MIN_SAMPLES_PER_CLASS:
            logger.warning(f"Imbalanced dataset: smallest class has only {min_class_count} sample(s)")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Combined dataset too imbalanced for stratified training",
                    "class_distribution": class_counts.to_dict(),
                    "minimum_per_class": MIN_SAMPLES_PER_CLASS,
                    "hint": "Need more balanced training data"
                }
            )

        # Step 9: Train models (one per specialty_group) with ALL data
        new_data_count = len(features)
        total_data_count = len(all_training_data)
        historical_data_count = total_data_count - new_data_count

        # Ensure specialty_group column exists (needed by train_model)
        if "specialty_group" not in all_training_data.columns:
            logger.info("Deriving specialty_group from specialty for loaded training data")
            from noshow_lib.feature_engineering import _create_specialty_group
            all_training_data = _create_specialty_group(all_training_data)

        trainer = ModelTrainer()
        training_output = trainer.train(features=all_training_data, config=config_dict)
        
        # Aggregate metrics across specialties
        all_metrics = {
            specialty: data["metrics"]
            for specialty, data in training_output.items()
        }
        all_thresholds = {
            specialty: data["threshold"]
            for specialty, data in training_output.items()
        }
        specialties_trained = list(training_output.keys())
        
        logger.info(
            f"Training completed for {len(specialties_trained)} specialties: {specialties_trained}"
        )
        
        # Step 10: Upload all specialty models to blob storage
        logger.info("Uploading trained specialty models to Azure Blob Storage")
        
        upload_results = None
        if not blob_service.is_configured():
            logger.warning("Azure Blob Storage not configured - skipping model upload")
        else:
            upload_results = blob_service.upload_specialty_models(
                training_output=training_output,
                environment=config.ENVIRONMENT,
            )
            
            if upload_results:
                logger.info(
                    f"All models uploaded successfully. "
                    f"Timestamp: {upload_results['timestamp']}"
                )
            else:
                logger.warning("Model upload failed")

        # Step 11: Save model history to database (one record per specialty)
        logger.info("Saving model histories to database")
        try:
            history_saver = ModelHistorySaver()
            history_saver.save_specialty_histories(
                db=db,
                training_output=training_output,
                upload_results=upload_results,
                environment=config.ENVIRONMENT,
                file_metadata={
                    "filename": file.filename,
                    "rows": validation_result.get("rows"),
                    "columns": validation_result.get("columns"),
                    "file_size_mb": validation_result.get("file_size_mb"),
                },
            )
            logger.info("Model histories saved")
        except Exception as e:
            logger.error(f"Failed to save model history: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save model history: {str(e)}"
            )

        # Build blob_urls summary for response
        blob_urls = None
        if upload_results and upload_results.get("specialties"):
            blob_urls = {
                specialty: sp_info.get("versioned_url")
                for specialty, sp_info in upload_results["specialties"].items()
            }

        response = TrainingResponse(
            status="success",
            message=f"Training completed for {len(specialties_trained)} specialty groups",
            specialties_trained=specialties_trained,
            blob_urls=blob_urls,
            metrics=all_metrics,
            thresholds=all_thresholds,
            training_time_seconds=sum(
                m.get("training_time_seconds", 0)
                for m in all_metrics.values()
                if isinstance(m, dict)
            ),
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

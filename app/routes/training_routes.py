import os
import shutil
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.config import config
from app.database import get_db, SessionLocal
from app.models.schemas import TrainingResponse, TrainingJobAccepted, TrainingJobStatus, ValidationResponse
from app.models.sql_models import TrainingJob
from app.services import (
    DataValidator,
    engineer_features,
    ModelTrainer,
    ModelHistorySaver,
    BlobStorageService
)
from app.services.appointment_feedback_service import load_feedback_as_features
from app.utils.logger import setup_logger
from app.utils.helpers import (
    generate_unique_filename,
    ensure_directory_exists,
    cleanup_temp_file
)

logger = setup_logger(__name__)
router = APIRouter(tags=["Training"])


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


@router.post("/upload-and-train", response_model=TrainingJobAccepted, status_code=202)
async def upload_and_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Data file (CSV, Excel, or Parquet)"),
    db: Session = Depends(get_db)
):
    """Upload file and queue model training. Returns immediately with a job_id to poll for status."""
    temp_file_path = None

    try:
        # Guard: reject if a training job is already in progress
        active_job = (
            db.query(TrainingJob)
            .filter(TrainingJob.status.in_(["pending", "running"]))
            .first()
        )
        if active_job:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A training job is already in progress. Wait for it to finish before starting a new one.",
                    "active_job_id": active_job.job_id,
                    "active_job_status": active_job.status,
                    "status_url": f"/training/status/{active_job.job_id}",
                },
            )

        temp_dir = ensure_directory_exists(config.UPLOAD_TEMP_DIR)
        unique_filename = generate_unique_filename(file.filename)
        temp_file_path = str(temp_dir / unique_filename)

        logger.info(f"Saving uploaded file to {temp_file_path}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            status="pending",
            original_filename=file.filename,
        )
        db.add(job)
        db.commit()

        background_tasks.add_task(
            _run_training_background,
            job_id=job_id,
            temp_file_path=temp_file_path,
            original_filename=file.filename,
        )

        status_url = f"/training/status/{job_id}"
        logger.info(f"Training job {job_id} queued. Poll: {status_url}")
        return TrainingJobAccepted(job_id=job_id, status_url=status_url)

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)
        logger.error(f"Failed to queue training job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to queue training job: {str(e)}")


@router.get("/status/{job_id}", response_model=TrainingJobStatus)
async def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """Poll the status of a training job previously queued by /upload-and-train."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")

    return TrainingJobStatus(
        job_id=job.job_id,
        status=job.status,
        result=job.result_json if job.status == "success" else None,
        error=job.error_message if job.status == "failed" else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _run_training_background(job_id: str, temp_file_path: str, original_filename: str) -> None:
    """Background task that executes the full training pipeline and updates job status in DB."""
    db: Session = SessionLocal()
    try:
        # Mark as running
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            logger.error(f"Background task: job {job_id} not found in DB")
            return
        job.status = "running"
        job.started_at = datetime.now()
        db.commit()

        # Step 1: Download config
        blob_service = BlobStorageService()
        try:
            config_dict = blob_service.get_config_from_blob()
            required_columns = config_dict.get("schema", {}).get("required_columns", [])
            logger.info(f"[Job {job_id}] Config loaded with {len(required_columns)} required columns")
        except Exception as e:
            raise RuntimeError(f"Failed to download configuration: {str(e)}")

        # Step 2: Basic validation
        logger.info(f"[Job {job_id}] Starting data validation")
        validator = DataValidator(required_columns=required_columns)
        validation_result, df = validator.load_and_validate(temp_file_path)

        if not validation_result["is_valid"]:
            raise ValueError(
                f"Data validation failed: {validation_result['errors']}"
            )
        logger.info(f"[Job {job_id}] Data validated. Shape: {df.shape}")

        # Step 3: Validate with noshow_lib schema
        try:
            validator.validate_with_config(df, config_dict)
        except ValueError as e:
            raise ValueError(f"Schema validation failed: {str(e)}")

        # Step 4: Generate batch ID and save raw data as Parquet to Blob Storage
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        df["batch_id"] = batch_id

        logger.info(f"[Job {job_id}] Persisting raw data as Parquet (batch={batch_id})")
        try:
            blob_service.upload_raw_parquet(
                df=df,
                environment=config.ENVIRONMENT,
                batch_label=batch_id,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save raw data to blob storage: {str(e)}")

        df["raw_appointment_id"] = range(len(df))

        # Step 5: Feature engineering
        logger.info(f"[Job {job_id}] Before build_features - Shape: {df.shape}")
        features = engineer_features(df, config_dict)
        logger.info(f"[Job {job_id}] After build_features - Shape: {features.shape}")

        # Step 6: Save training data as Parquet
        logger.info(f"[Job {job_id}] Persisting training data as Parquet")
        try:
            blob_service.upload_training_parquet(
                df=features,
                environment=config.ENVIRONMENT,
                batch_label=batch_id,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save training data to blob storage: {str(e)}")

        # Step 7: Load ALL training data (historical + new)
        logger.info(f"[Job {job_id}] Loading all training data from Parquet files")
        try:
            all_training_data = blob_service.load_all_training_parquets(
                environment=config.ENVIRONMENT,
                limit=config.TRAINING_DATA_LIMIT,
                days_lookback=config.TRAINING_DAYS_LOOKBACK,
            )
            logger.info(f"[Job {job_id}] Loaded {len(all_training_data)} total rows")
        except Exception as e:
            raise RuntimeError(f"Failed to load training data from blob storage: {str(e)}")

        # Step 7B: Enrich with feedback from appointment_predictions
        logger.info(f"[Job {job_id}] Enriching dataset with feedback")
        try:
            feedback_features = load_feedback_as_features(db=db, config_dict=config_dict)
            if not feedback_features.empty:
                parquet_rows = len(all_training_data)
                all_training_data = pd.concat(
                    [all_training_data, feedback_features], ignore_index=True
                )
                logger.info(
                    f"[Job {job_id}] Dataset enriched: {parquet_rows} Parquet + "
                    f"{len(feedback_features)} feedback = {len(all_training_data)} total"
                )
            else:
                logger.info(f"[Job {job_id}] No feedback records — training on Parquet data only")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Feedback enrichment failed: {e}. Proceeding without it.")

        # Step 8: Validate combined dataset size
        MIN_SAMPLES_TOTAL = 20
        MIN_SAMPLES_PER_CLASS = 2
        target_col = config_dict.get("data", {}).get("target_column", "no_show")

        if target_col not in all_training_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")

        class_counts = all_training_data[target_col].value_counts()
        total_samples = len(all_training_data)
        min_class_count = class_counts.min() if len(class_counts) > 0 else 0

        if total_samples < MIN_SAMPLES_TOTAL:
            raise ValueError(
                f"Combined dataset too small: {total_samples} samples (minimum: {MIN_SAMPLES_TOTAL})"
            )
        if min_class_count < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Dataset too imbalanced: smallest class has {min_class_count} sample(s) "
                f"(minimum: {MIN_SAMPLES_PER_CLASS})"
            )

        # Step 9: Train models (one per specialty_group)
        if "specialty_group" not in all_training_data.columns:
            logger.info(f"[Job {job_id}] Deriving specialty_group from specialty")
            from noshow_lib.feature_engineering import _create_specialty_group
            all_training_data = _create_specialty_group(all_training_data)

        trainer = ModelTrainer()
        training_output = trainer.train(features=all_training_data, config=config_dict)

        all_metrics = {sp: data["metrics"] for sp, data in training_output.items()}
        all_thresholds = {sp: data["threshold"] for sp, data in training_output.items()}
        specialties_trained = list(training_output.keys())

        logger.info(f"[Job {job_id}] Training completed for {len(specialties_trained)} specialties")

        # Step 10: Upload models to blob storage
        upload_results = None
        if not blob_service.is_configured():
            logger.warning(f"[Job {job_id}] Blob Storage not configured — skipping model upload")
        else:
            upload_results = blob_service.upload_specialty_models(
                training_output=training_output,
                environment=config.ENVIRONMENT,
            )

        # Step 11: Save model history
        logger.info(f"[Job {job_id}] Saving model histories to database")
        try:
            history_saver = ModelHistorySaver()
            history_saver.save_specialty_histories(
                db=db,
                training_output=training_output,
                upload_results=upload_results,
                environment=config.ENVIRONMENT,
                file_metadata={
                    "filename": original_filename,
                    "rows": validation_result.get("rows"),
                    "columns": validation_result.get("columns"),
                    "file_size_mb": validation_result.get("file_size_mb"),
                },
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save model history: {str(e)}")

        blob_urls = None
        if upload_results and upload_results.get("specialties"):
            blob_urls = {
                sp: sp_info.get("versioned_url")
                for sp, sp_info in upload_results["specialties"].items()
            }

        training_time = sum(
            m.get("training_time_seconds", 0)
            for m in all_metrics.values()
            if isinstance(m, dict)
        )

        # Mark job as success
        job.status = "success"
        job.finished_at = datetime.now()
        job.result_json = {
            "status": "success",
            "message": f"Training completed for {len(specialties_trained)} specialty groups",
            "specialties_trained": specialties_trained,
            "blob_urls": blob_urls,
            "metrics": all_metrics,
            "thresholds": all_thresholds,
            "training_time_seconds": training_time,
        }
        db.commit()
        logger.info(f"[Job {job_id}] Job completed successfully")

    except Exception as e:
        logger.error(f"[Job {job_id}] Training pipeline failed: {str(e)}", exc_info=True)
        try:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.finished_at = datetime.now()
                db.commit()
        except Exception as db_err:
            logger.error(f"[Job {job_id}] Failed to update job status in DB: {db_err}")
    finally:
        db.close()
        if os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)

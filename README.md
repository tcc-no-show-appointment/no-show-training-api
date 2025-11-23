# No-Show Training API

A FastAPI application for training machine learning models to predict no-show appointments, integrated with the `noshow_lib` library.

## Status

**Fully functional implementation** using the `noshow_lib` package for ML training pipeline.

## Features

### Implemented

- ✅ File validation (CSV, Excel, Parquet)
- ✅ Azure Blob Storage integration
- ✅ RESTful API endpoints
- ✅ Comprehensive logging
- ✅ Error handling
- ✅ Data preprocessing via `noshow_lib`
- ✅ Feature engineering via `noshow_lib`
- ✅ Model training via `noshow_lib` (external access mode)
- ✅ Metrics evaluation (ROC-AUC, F1, Recall, Precision, Accuracy)
- ✅ Automatic model serialization to bytes for blob upload

### TODO

- ⏳ Model history persistence (database integration)

## API Endpoints

### `POST /training/validate`

Validates an uploaded file without processing.

**Request:**

- File upload (multipart/form-data)
- Supported formats: CSV, Excel (.xlsx, .xls), Parquet

**Response:**

```json
{
  "is_valid": true,
  "file_format": "csv",
  "file_size_mb": 2.5,
  "rows": 10000,
  "columns": 14,
  "missing_columns": [],
  "errors": [],
  "warnings": []
}
```

### `POST /training/upload-and-train`

Executes the complete training flow using the `noshow_lib` pipeline.

**Flow:**

1. Validate uploaded file
2. Preprocess data using `noshow_lib.load_and_process_data()`
3. Feature engineering using `noshow_lib.build_features()`
4. Model training using `noshow_lib.train_model()` (external access mode)
5. Automatic model serialization to joblib bytes
6. Upload trained model to Azure Blob Storage
7. Return metrics and blob URL

**Request:**

- File upload (multipart/form-data)

**Response:**

```json
{
  "status": "success",
  "message": "Training flow completed successfully",
  "model_filename": "noshow_model_20250123_143022.joblib",
  "blob_url": "https://storage.blob.core.windows.net/models/noshow_model_20250123_143022.joblib",
  "metrics": {
    "threshold_used": 0.3456,
    "roc_auc": 0.8234,
    "average_precision": 0.7891,
    "f1": 0.6543,
    "recall": 0.7123,
    "precision": 0.6012,
    "accuracy": 0.8456
  },
  "training_time_seconds": 45.3,
  "timestamp": "2025-11-23T14:30:22.000000"
}
```

### `GET /`

Health check endpoint.

### `GET /info`

API information and available endpoints.

## Required File Columns

The uploaded CSV/Excel/Parquet file must contain the following columns:

- `PatientId`
- `AppointmentID`
- `Gender`
- `ScheduledDay`
- `AppointmentDay`
- `Age`
- `Neighbourhood`
- `Scholarship`
- `Hipertension`
- `Diabetes`
- `Alcoholism`
- `Handcap`
- `SMS_received`
- `No-show`

## Installation

```bash
# Clone the repository
git clone https://github.com/tcc-no-show-appointment/no-show-training-api.git
cd no-show-training-api

# Install dependencies (including noshow_lib)
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure credentials
```

## Dependencies

This API uses the `noshow_lib` package for ML operations:

```bash
# noshow_lib is automatically installed from requirements.txt
# It provides:
# - load_and_process_data: Data preprocessing
# - build_features: Feature engineering
# - train_model: Model training with external access mode
```

## Environment Variables

```env
# Azure Storage
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_ACCOUNT_KEY=your_account_key
AZURE_BLOB_CONTAINER_NAME=your_container

# API Configuration
MAX_FILE_SIZE_MB=100
UPLOAD_TEMP_DIR=temp_uploads

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
```

## Running the API

```bash
# Development
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
app/
├── __init__.py
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── constants.py           # Constants and validation rules
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic models
├── routes/
│   ├── __init__.py
│   └── training_routes.py # Training endpoints
├── services/
│   ├── __init__.py
│   ├── data_validation.py      # File validation
│   ├── data_preprocessing.py   # Preprocessing (uses noshow_lib)
│   ├── training_service.py     # Training (uses noshow_lib)
│   ├── model_history.py        # History tracking (TODO)
│   └── blob_service.py         # Azure Blob Storage
└── utils/
    ├── __init__.py
    ├── logger.py          # Logging configuration
    └── helpers.py         # Helper functions
```

## How It Works

### Training Pipeline

The API uses `noshow_lib` in **external access mode**, which means:

1. **No file I/O** - All operations happen in memory
2. **Auto-configuration** - Model parameters loaded from `noshow_lib/config.yaml`
3. **Serialized output** - Returns model as joblib bytes ready for blob upload

**Code Flow:**

```python
# 1. Preprocess data
from noshow_lib import load_and_process_data
processed_data = load_and_process_data(raw_dataframe)

# 2. Engineer features
from noshow_lib import build_features
features = build_features(processed_data, {"target_column": "No-show"})

# 3. Train model (external access mode)
from noshow_lib import train_model
model_bytes, metrics = train_model(
    df_input=features,
    is_external_access=True  # Returns bytes instead of saving to disk
)

# 4. Upload to blob
blob_service.upload_bytes(model_bytes, blob_name="model.joblib")
```

### External Access Mode Benefits

- ✅ **Memory-only operations** - Perfect for APIs and microservices
- ✅ **No manual serialization** - Model already in joblib bytes format
- ✅ **Direct blob upload** - No intermediate file storage needed
- ✅ **Automatic config** - Uses `noshow_lib`'s configuration automatically

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## Docker

```bash
# Build image
docker build -t no-show-training-api .

# Run container
docker run -p 8000:8000 --env-file .env no-show-training-api
```

## Next Steps

To enhance the API:

1. **Implement model history tracking**: Add database integration for version control
2. **Add model versioning**: Track different model versions with metadata
3. **Implement A/B testing**: Support for deploying multiple model versions
4. **Add monitoring**: Track prediction performance over time
5. **Enhance validation**: Add more sophisticated data quality checks

## Model Configuration

The ML model configuration is managed by `noshow_lib`. To customize:

1. Clone the `noshow_lib` repository
2. Edit `config.yaml` with your desired parameters:
   - Model type (LGBMClassifier, RandomForest, etc.)
   - Hyperparameters
   - Feature engineering rules
   - Class balancing settings (SMOTE)
   - Evaluation metrics

## Metrics Returned

The API returns comprehensive evaluation metrics:

- `roc_auc`: Area under the ROC curve
- `average_precision`: Average precision score
- `f1`: F1 score (harmonic mean of precision and recall)
- `recall`: Recall (sensitivity)
- `precision`: Precision
- `accuracy`: Overall accuracy
- `threshold_used`: Optimized decision threshold

## License

MIT

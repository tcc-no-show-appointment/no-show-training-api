REQUIRED_COLUMNS = [
    "PatientId",
    "AppointmentID",
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "No-show"
]

NUMERICAL_FEATURES = [
    "Age",
    "waiting_days",
    "scheduled_hour",
    "weekday_sin",
    "weekday_cos",
    "wait_x_sms"
]

CATEGORICAL_LOW_CARDINALITY = [
    "Gender",
    "waiting_days_bucket"
]

CATEGORICAL_HIGH_CARDINALITY = [
    "Neighbourhood"
]

BINARY_FEATURES = [
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "is_weekend",
    "day_part_morning",
    "day_part_afternoon",
    "day_part_evening"
]

DROP_COLUMNS = [
    "No-show",
    "PatientId",
    "AppointmentID",
    "ScheduledDay",
    "AppointmentDay"
]

WAITING_DAYS_BINS = [-1, 0, 3, 7, 14, 10**9]
WAITING_DAYS_LABELS = ["wait_0", "wait_1_3", "wait_4_7", "wait_8_14", "wait_15p"]

MIN_AGE = 0
MAX_AGE = 100

MODEL_FILE_PREFIX = "rf_grid_model"
MODEL_FILE_EXTENSION = ".pkl"

STATUS_MESSAGES = {
    "VALIDATION_ERROR": "File validation failed",
    "PREPROCESSING_ERROR": "Data preprocessing failed",
    "TRAINING_ERROR": "Model training failed",
    "UPLOAD_ERROR": "Failed to upload model to Azure Blob Storage",
    "SUCCESS": "Model training completed successfully"
}

SUPPORTED_FORMATS = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "parquet": "application/octet-stream"
}

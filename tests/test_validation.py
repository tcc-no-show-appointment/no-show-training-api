import sys
import os
import pytest
import pandas as pd
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.data_validation import DataValidator


def test_load_and_validate():
    """Test load_and_validate with a real CSV file"""
    df = pd.DataFrame([{
        'PatientId': 123,
        'AppointmentID': 456,
        'Gender': 'F',
        'Age': 62,
        'ScheduledDay': '2024-11-16T08:00:00',
        'AppointmentDay': '2024-11-23T14:00:00',
        'Neighbourhood': 'JARDIM CAMBURI',
        'Scholarship': 0,
        'Hipertension': 1,
        'Diabetes': 0,
        'Alcoholism': 0,
        'Handcap': 0,
        'SMS_received': 1,
        'No-show': 'No'
    }])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        validator = DataValidator()
        result, loaded_df = validator.load_and_validate(temp_path)
        
        assert result['is_valid']
        assert loaded_df is not None
        assert len(loaded_df) == 1
    finally:
        os.unlink(temp_path)

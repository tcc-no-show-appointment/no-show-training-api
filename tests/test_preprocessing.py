import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.data_preprocessing import process_raw_data, engineer_features


def test_process_raw_data():
    """Test real data processing with noshow_lib"""
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
    
    result = process_raw_data(df)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_engineer_features():
    """Test real feature engineering with noshow_lib"""
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
    
    processed = process_raw_data(df)
    features = engineer_features(processed)
    
    assert features is not None
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0

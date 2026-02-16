import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import (
    create_time_features,
    create_transaction_frequency_features,
    create_aggregated_features
)

@pytest.fixture
def sample_fraud_data():
    data = {
        'user_id': [1, 1, 2, 3],
        'signup_time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00']),
        'purchase_time': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 13:00:00', '2023-01-02 13:00:00', '2023-01-03 14:30:00']),
        'purchase_value': [100, 50, 200, 150],
        'age': [25, 25, 30, 35]
    }
    return pd.DataFrame(data)

def test_create_time_features(sample_fraud_data):
    df_feat = create_time_features(sample_fraud_data)
    
    assert 'hour_of_day' in df_feat.columns
    assert 'day_of_week' in df_feat.columns
    assert 'time_since_signup' in df_feat.columns
    
    # Check specific values
    # First user: purchase at 12:00, signup at 10:00 -> 2 hours diff
    assert df_feat.loc[0, 'time_since_signup'] == 2.0
    assert df_feat.loc[0, 'hour_of_day'] == 12

def test_create_transaction_frequency_features(sample_fraud_data):
    df_feat = create_transaction_frequency_features(sample_fraud_data, time_windows=[1])
    
    assert 'transactions_in_1h' in df_feat.columns
    assert 'transaction_velocity' in df_feat.columns
    
    # Check logic: user 1 has 2 transactions.
    # 2nd transaction (idx 1) is at 13:00, 1st (idx 0) is at 12:00.
    # Within 1h window of 13:00 (12:00-13:00), both serve? 
    # Let's check the logic in source: 
    # window_start = current_time - window_seconds
    # 13:00 - 1h = 12:00. 12:00 is >= 12:00. So count should be 2 for the second transaction.
    
    # Wait, the logic is: sum(purchase_times[:i+1] >= window_start)
    # For index 1 (13:00): window_start is 12:00.
    # purchase_times[:2] are 12:00 and 13:00. Both are >= 12:00. So count is 2.
    assert df_feat.loc[1, 'transactions_in_1h'] == 2

def test_create_aggregated_features(sample_fraud_data):
    df_feat = create_aggregated_features(sample_fraud_data)
    
    expected_cols = [
        'user_purchase_value_mean',
        'user_purchase_value_sum',
        'user_age_mean' # mean
    ]
    
    for col in expected_cols:
        assert col in df_feat.columns
        
    # Check aggregation
    # User 1: values 100, 50 -> mean 75, sum 150
    user1_rows = df_feat[df_feat['user_id'] == 1]
    assert user1_rows['user_purchase_value_mean'].iloc[0] == 75.0
    assert user1_rows['user_purchase_value_sum'].iloc[0] == 150.0

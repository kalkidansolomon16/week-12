"""
Feature engineering utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from typing import List


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with signup_time and purchase_time columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional time-based features
    """
    df_feat = df.copy()
    
    if 'purchase_time' in df_feat.columns:
        df_feat['hour_of_day'] = df_feat['purchase_time'].dt.hour
        df_feat['day_of_week'] = df_feat['purchase_time'].dt.dayofweek
        df_feat['day_of_month'] = df_feat['purchase_time'].dt.day
        df_feat['month'] = df_feat['purchase_time'].dt.month
    
    if 'signup_time' in df_feat.columns and 'purchase_time' in df_feat.columns:
        # Time since signup in hours
        df_feat['time_since_signup'] = (
            (df_feat['purchase_time'] - df_feat['signup_time']).dt.total_seconds() / 3600
        )
        # Handle negative values (shouldn't happen, but just in case)
        df_feat['time_since_signup'] = df_feat['time_since_signup'].clip(lower=0)
    
    return df_feat


def create_transaction_frequency_features(df: pd.DataFrame, 
                                         time_windows: List[int] = [1, 6, 24]) -> pd.DataFrame:
    """
    Create transaction frequency and velocity features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with user_id and purchase_time columns
    time_windows : List[int]
        List of time windows in hours for calculating frequency
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional frequency features
    """
    df_feat = df.copy()
    
    if 'user_id' not in df_feat.columns or 'purchase_time' not in df_feat.columns:
        return df_feat
    
    # Sort by user_id and purchase_time
    df_feat = df_feat.sort_values(['user_id', 'purchase_time'])
    
    # For each time window, calculate transactions per user
    for window_hours in time_windows:
        window_seconds = window_hours * 3600
        
        # Calculate transactions in the time window for each user
        def count_transactions_in_window(group):
            purchase_times = group['purchase_time'].values
            counts = []
            for i, current_time in enumerate(purchase_times):
                window_start = pd.Timestamp(current_time) - pd.Timedelta(seconds=window_seconds)
                count = np.sum(purchase_times[:i+1] >= window_start)
                counts.append(count)
            return counts
        
        freq_col = f'transactions_in_{window_hours}h'
        df_feat[freq_col] = df_feat.groupby('user_id').apply(
            lambda x: pd.Series(count_transactions_in_window(x), index=x.index)
        ).values
    
    # Calculate transaction velocity (transactions per hour for each user)
    df_feat['user_transaction_count'] = df_feat.groupby('user_id').cumcount() + 1
    df_feat['user_first_transaction'] = df_feat.groupby('user_id')['purchase_time'].transform('min')
    df_feat['user_account_age_hours'] = (
        (df_feat['purchase_time'] - df_feat['user_first_transaction']).dt.total_seconds() / 3600
    )
    df_feat['user_account_age_hours'] = df_feat['user_account_age_hours'].replace(0, 1)  # Avoid division by zero
    df_feat['transaction_velocity'] = df_feat['user_transaction_count'] / df_feat['user_account_age_hours']
    
    return df_feat


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features per user.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with user_id and other columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with aggregated features
    """
    df_feat = df.copy()
    
    if 'user_id' not in df_feat.columns:
        return df_feat
    
    # User-level aggregations
    user_stats = df_feat.groupby('user_id').agg({
        'purchase_value': ['mean', 'std', 'min', 'max', 'sum'],
        'age': 'mean',
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['user_id'] + [f'user_{col[0]}_{col[1]}' if col[1] else f'user_{col[0]}' 
                                        for col in user_stats.columns[1:]]
    
    # Merge back
    df_feat = df_feat.merge(user_stats, on='user_id', how='left')
    
    # Fill NaN values in aggregated columns
    for col in user_stats.columns:
        if col != 'user_id':
            df_feat[col].fillna(0, inplace=True)
    
    return df_feat



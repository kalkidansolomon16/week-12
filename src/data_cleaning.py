"""
Data cleaning utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Fraud_Data.csv dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw fraud data dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Convert timestamp columns to datetime
    if 'signup_time' in df_clean.columns:
        df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
    if 'purchase_time' in df_clean.columns:
        df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df_clean.isnull().sum()
    if missing_before.sum() > 0:
        print("\nMissing values before cleaning:")
        print(missing_before[missing_before > 0])
        
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
        
        missing_after = df_clean.isnull().sum()
        print("\nMissing values after cleaning:")
        print(missing_after[missing_after > 0])
    
    # Ensure correct data types
    if 'age' in df_clean.columns:
        df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce').astype('Int64')
    if 'purchase_value' in df_clean.columns:
        df_clean['purchase_value'] = pd.to_numeric(df_clean['purchase_value'], errors='coerce')
    
    return df_clean


def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the creditcard.csv dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw credit card data dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df_clean.isnull().sum()
    if missing_before.sum() > 0:
        print("\nMissing values before cleaning:")
        print(missing_before[missing_before > 0])
        
        # Fill numerical columns with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        missing_after = df_clean.isnull().sum()
        print("\nMissing values after cleaning:")
        print(missing_after[missing_after > 0])
    
    return df_clean


def ip_to_integer(ip_address: str) -> int:
    """
    Convert IP address string to integer.
    
    Parameters:
    -----------
    ip_address : str
        IP address in string format (e.g., '192.168.1.1')
    
    Returns:
    --------
    int
        IP address as integer
    """
    try:
        parts = ip_address.split('.')
        if len(parts) != 4:
            return None
        return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
    except:
        return None


def merge_ip_to_country(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud data with IP to country mapping using range-based lookup.
    
    Parameters:
    -----------
    fraud_df : pd.DataFrame
        Fraud data with ip_address column
    ip_country_df : pd.DataFrame
        IP to country mapping with lower_bound, upper_bound, and country columns
    
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with country information
    """
    # Convert IP addresses to integers
    fraud_df = fraud_df.copy()
    fraud_df['ip_integer'] = fraud_df['ip_address'].apply(ip_to_integer)
    
    # Sort both dataframes for merge_asof
    ip_country_sorted = ip_country_df.sort_values('lower_bound_ip_address')
    fraud_sorted = fraud_df.sort_values('ip_integer')
    
    # Use merge_asof for range-based lookup
    merged = pd.merge_asof(
        fraud_sorted,
        ip_country_sorted,
        left_on='ip_integer',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter to keep only rows where IP falls within the range
    merged = merged[
        (merged['ip_integer'] >= merged['lower_bound_ip_address']) &
        (merged['ip_integer'] <= merged['upper_bound_ip_address'])
    ]
    
    return merged



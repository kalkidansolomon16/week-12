"""
Data transformation utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Optional, List


def encode_categorical_features(df: pd.DataFrame, 
                                categorical_cols: List[str],
                                method: str = 'onehot',
                                encoders: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with categorical columns
    categorical_cols : List[str]
        List of categorical column names
    method : str
        Encoding method: 'onehot' or 'label'
    encoders : Optional[dict]
        Pre-fitted encoders (for transforming test data)
    
    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        Encoded dataframe and encoder objects dictionary
    """
    df_encoded = df.copy()
    if encoders is None:
        encoders = {}
    
    if method == 'onehot':
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col in encoders and hasattr(encoders[col], 'categories_'):
                    # Use existing encoder (for test set)
                    encoder = encoders[col]
                    encoded = encoder.transform(df_encoded[[col]])
                else:
                    # Create new encoder (for training set)
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded = encoder.fit_transform(df_encoded[[col]])
                    encoders[col] = encoder
                
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f'{col}_{cat}' for cat in encoder.categories_[0][1:]],
                    index=df_encoded.index
                )
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
    elif method == 'label':
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col in encoders:
                    # Use existing encoder (for test set)
                    encoder = encoders[col]
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                else:
                    # Create new encoder (for training set)
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    encoders[col] = encoder
    
    return df_encoded, encoders


def scale_numerical_features(df: pd.DataFrame,
                             numerical_cols: List[str],
                             scaler_type: str = 'standard',
                             fit: bool = True,
                             scaler: Optional[object] = None) -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with numerical columns
    numerical_cols : List[str]
        List of numerical column names
    scaler_type : str
        Type of scaler: 'standard' or 'minmax'
    fit : bool
        Whether to fit the scaler (True for training, False for testing)
    scaler : Optional[object]
        Pre-fitted scaler (required if fit=False)
    
    Returns:
    --------
    Tuple[pd.DataFrame, object]
        Scaled dataframe and scaler object
    """
    df_scaled = df.copy()
    
    if scaler_type == 'standard':
        if fit or scaler is None:
            scaler = StandardScaler()
        else:
            scaler = scaler
    elif scaler_type == 'minmax':
        if fit or scaler is None:
            scaler = MinMaxScaler()
        else:
            scaler = scaler
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Filter to only columns that exist
    existing_cols = [col for col in numerical_cols if col in df_scaled.columns]
    
    if existing_cols:
        if fit:
            df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
        else:
            df_scaled[existing_cols] = scaler.transform(df_scaled[existing_cols])
    
    return df_scaled, scaler


def handle_class_imbalance(X: pd.DataFrame,
                           y: pd.Series,
                           method: str = 'smote',
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE or undersampling.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    method : str
        Method to use: 'smote' or 'undersample'
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X and y
    """
    print(f"\nClass distribution before resampling:")
    print(y.value_counts())
    print(f"Class distribution (%):")
    print(y.value_counts(normalize=True) * 100)
    
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)
    
    print(f"\nClass distribution after {method}:")
    print(y_resampled.value_counts())
    print(f"Class distribution (%):")
    print(y_resampled.value_counts(normalize=True) * 100)
    
    return X_resampled, y_resampled


def prepare_data_for_modeling(df: pd.DataFrame,
                              target_col: str,
                              categorical_cols: List[str],
                              numerical_cols: List[str],
                              test_size: float = 0.2,
                              random_state: int = 42,
                              handle_imbalance: bool = True,
                              imbalance_method: str = 'smote') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Prepare data for modeling: split, encode, scale, and handle imbalance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    target_col : str
        Name of target column
    categorical_cols : List[str]
        List of categorical column names
    numerical_cols : List[str]
        List of numerical column names
    test_size : float
        Proportion of data for testing
    random_state : int
        Random state for reproducibility
    handle_imbalance : bool
        Whether to handle class imbalance
    imbalance_method : str
        Method for handling imbalance: 'smote' or 'undersample'
    
    Returns:
    --------
    Tuple containing:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target
    - y_test: Testing target
    - transformers: Dictionary of transformers (encoders, scalers)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    transformers = {}
    
    # Encode categorical features
    if categorical_cols:
        X_train_encoded, encoders = encode_categorical_features(X_train, categorical_cols)
        X_test_encoded, _ = encode_categorical_features(X_test, categorical_cols, encoders=encoders)
        transformers['encoders'] = encoders
    else:
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
    
    # Scale numerical features
    if numerical_cols:
        X_train_scaled, scaler = scale_numerical_features(
            X_train_encoded, numerical_cols, fit=True
        )
        X_test_scaled, _ = scale_numerical_features(
            X_test_encoded, numerical_cols, fit=False, scaler=scaler
        )
        transformers['scaler'] = scaler
    else:
        X_train_scaled = X_train_encoded.copy()
        X_test_scaled = X_test_encoded.copy()
    
    # Handle class imbalance on training data only
    if handle_imbalance:
        X_train_resampled, y_train_resampled = handle_class_imbalance(
            X_train_scaled, y_train, method=imbalance_method, random_state=random_state
        )
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test, transformers
    else:
        return X_train_scaled, X_test_scaled, y_train, y_test, transformers


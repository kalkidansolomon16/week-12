"""
Modeling utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def train_baseline_model(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        random_state: int = 42) -> Tuple[LogisticRegression, Dict]:
    """
    Train a Logistic Regression baseline model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    Tuple[LogisticRegression, Dict]
        Trained model and evaluation metrics
    """
    print("Training Logistic Regression baseline model...")
    
    # Train model
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'  # Handle imbalance
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': model.score(X_train, y_train),
        'train_roc_auc': roc_auc_score(y_train, y_pred_proba),
        'train_ap': average_precision_score(y_train, y_pred_proba),
        'train_f1': f1_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred),
        'train_recall': recall_score(y_train, y_pred)
    }
    
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Training ROC-AUC: {metrics['train_roc_auc']:.4f}")
    print(f"Training AP (AUC-PR): {metrics['train_ap']:.4f}")
    print(f"Training F1-Score: {metrics['train_f1']:.4f}")
    
    return model, metrics


def train_ensemble_model(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         model_type: str = 'random_forest',
                         random_state: int = 42,
                         n_estimators: int = 100,
                         max_depth: int = 10) -> Tuple[object, Dict]:
    """
    Train an ensemble model (Random Forest, XGBoost, or LightGBM).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_type : str
        Type of ensemble: 'random_forest', 'xgboost', or 'lightgbm'
    random_state : int
        Random state for reproducibility
    n_estimators : int
        Number of estimators
    max_depth : int
        Maximum depth of trees
    
    Returns:
    --------
    Tuple[object, Dict]
        Trained model and evaluation metrics
    """
    print(f"Training {model_type} ensemble model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': model.score(X_train, y_train),
        'train_roc_auc': roc_auc_score(y_train, y_pred_proba),
        'train_ap': average_precision_score(y_train, y_pred_proba),
        'train_f1': f1_score(y_train, y_pred),
        'train_precision': precision_score(y_train, y_pred),
        'train_recall': recall_score(y_train, y_pred)
    }
    
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Training ROC-AUC: {metrics['train_roc_auc']:.4f}")
    print(f"Training AP (AUC-PR): {metrics['train_ap']:.4f}")
    print(f"Training F1-Score: {metrics['train_f1']:.4f}")
    
    return model, metrics


def evaluate_model(model: object,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   model_name: str = "Model") -> Dict:
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    Dict
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {model_name} on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'test_accuracy': model.score(X_test, y_test),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_ap': average_precision_score(y_test, y_pred_proba),
        'test_f1': f1_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"Test AP (AUC-PR): {metrics['test_ap']:.4f}")
    print(f"Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"Test Precision: {metrics['test_precision']:.4f}")
    print(f"Test Recall: {metrics['test_recall']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics


def cross_validate_model(X: pd.DataFrame,
                         y: pd.Series,
                         model: object,
                         cv: int = 5,
                         scoring: List[str] = ['roc_auc', 'average_precision', 'f1']) -> Dict:
    """
    Perform stratified K-fold cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model : object
        Model to cross-validate
    cv : int
        Number of folds
    scoring : List[str]
        List of scoring metrics
    
    Returns:
    --------
    Dict
        Dictionary of cross-validation results
    """
    print(f"\nPerforming {cv}-fold Stratified Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    
    for score_name in scoring:
        if score_name == 'roc_auc':
            scorer = 'roc_auc'
        elif score_name == 'average_precision':
            scorer = 'average_precision'
        elif score_name == 'f1':
            scorer = 'f1'
        else:
            continue
        
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        results[score_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"{score_name.upper()}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results


def save_model(model: object, filepath: str):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : object
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    
    Returns:
    --------
    object
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


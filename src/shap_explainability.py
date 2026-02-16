"""
SHAP explainability utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract built-in feature importance from ensemble model.
    
    Parameters:
    -----------
    model : object
        Trained ensemble model (Random Forest, XGBoost, etc.)
    feature_names : List[str]
        List of feature names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df


def plot_feature_importance(feature_importance_df: pd.DataFrame, 
                           top_n: int = 10,
                           title: str = "Top Feature Importance"):
    """
    Plot top N feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with feature importance
    top_n : int
        Number of top features to plot
    title : str
        Plot title
    """
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def create_shap_explainer(model, X_sample: pd.DataFrame, model_type: str = 'tree'):
    """
    Create SHAP explainer for the model.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_sample : pd.DataFrame
        Sample of data for SHAP (typically 100-1000 samples)
    model_type : str
        Type of model: 'tree', 'linear', or 'kernel'
    
    Returns:
    --------
    shap.Explainer
        SHAP explainer object
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    return explainer


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, max_display: int = 20):
    """
    Plot SHAP summary plot (global feature importance).
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values
    X_sample : pd.DataFrame
        Sample data
    max_display : int
        Maximum number of features to display
    """
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_force_plot(explainer, X_instance: pd.DataFrame, 
                        instance_idx: int, expected_value: float):
    """
    Plot SHAP force plot for individual prediction.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer
    X_instance : pd.DataFrame
        Instance data
    instance_idx : int
        Index of instance to explain
    expected_value : float
        Expected value (base value) from explainer
    """
    shap_values_instance = explainer.shap_values(X_instance.iloc[[instance_idx]])
    
    if isinstance(shap_values_instance, list):
        # For binary classification, use positive class
        shap_values_instance = shap_values_instance[1]
    
    shap.force_plot(
        expected_value,
        shap_values_instance[0],
        X_instance.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.show()


def get_top_fraud_drivers(shap_values: np.array, 
                         feature_names: List[str], 
                         top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Identify top drivers of fraud predictions from SHAP values.
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values
    feature_names : List[str]
        List of feature names
    top_n : int
        Number of top drivers to return
    
    Returns:
    --------
    List[Tuple[str, float]]
        List of (feature_name, mean_abs_shap_value) tuples
    """
    if len(shap_values.shape) > 2:
        # For binary classification, use positive class
        shap_values = shap_values[1]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance[:top_n]


import pandas as pd
import numpy as np
import pytest
from src.modeling import (
    train_baseline_model,
    train_ensemble_model,
    evaluate_model,
    cross_validate_model,
    save_model,
    load_model
)
from sklearn.linear_model import LogisticRegression
import os

@pytest.fixture
def synthetic_data():
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(np.random.randn(n_samples, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y

def test_train_baseline_model(synthetic_data):
    X, y = synthetic_data
    model, metrics = train_baseline_model(X, y)
    
    assert isinstance(model, LogisticRegression)
    assert 'train_accuracy' in metrics
    assert metrics['train_roc_auc'] >= 0.0

def test_train_random_forest(synthetic_data):
    X, y = synthetic_data
    # Use fewer estimators to speed up test
    model, metrics = train_ensemble_model(X, y, model_type='random_forest', n_estimators=10)
    
    assert 'train_accuracy' in metrics
    assert metrics['train_f1'] >= 0.0

def test_evaluate_model(synthetic_data):
    X, y = synthetic_data
    model, _ = train_baseline_model(X, y)
    
    metrics = evaluate_model(model, X, y, model_name="Test Model")
    
    assert 'test_accuracy' in metrics
    assert 'confusion_matrix' in metrics

def test_save_and_load_model(synthetic_data, tmp_path):
    X, y = synthetic_data
    model, _ = train_baseline_model(X, y)
    
    filepath = tmp_path / "test_model.pkl"
    save_model(model, str(filepath))
    
    assert os.path.exists(filepath)
    
    loaded_model = load_model(str(filepath))
    assert isinstance(loaded_model, LogisticRegression)
    
    # Check predictions match
    pred1 = model.predict(X)
    pred2 = loaded_model.predict(X)
    assert np.array_equal(pred1, pred2)

# Fraud Detection Project

## Overview
This project focuses on improving the detection of fraud cases for e-commerce and bank transactions using advanced machine learning models and detailed data analysis.

## Project Structure
```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                      # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── models/                      # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your data files in `data/raw/`:
   - `Fraud_Data.csv`
   - `IpAddress_to_Country.csv`
   - `creditcard.csv`
5. Run the notebooks in order:
   - `notebooks/eda-fraud-data.ipynb`
   - `notebooks/eda-creditcard.ipynb`
   - `notebooks/feature-engineering.ipynb`
   - `notebooks/modeling.ipynb`
   - `notebooks/shap-explainability.ipynb`

## Tasks

### Task 1 - Data Analysis and Preprocessing
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Geolocation integration
- Feature engineering
- Data transformation
- Class imbalance handling

### Task 2 - Model Building and Training
- Baseline model (Logistic Regression)
- Ensemble model (Random Forest/XGBoost/LightGBM)
- Model evaluation and comparison

### Task 3 - Model Explainability
- SHAP analysis
- Feature importance
- Business recommendations

## Author
Data Scientist at Adey Innovations Inc.


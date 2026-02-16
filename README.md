# 🛡️ Fraud Detection Analytics System

Understanding and preventing digital transaction fraud through advanced data science.

## 📌 Business Problem

E-commerce platforms and financial institutions lose billions annually to fraudulent transactions.

- **Challenge**: Detecting fraudulent user behavior in real-time without compromising the user experience for legitimate customers.
- **Goal**: Develop a robust machine learning system to classify transactions as legitimate or fraudulent, minimizing false positives while maximizing fraud detection.

## 🚀 Solution Overview

This project implements an end-to-end fraud detection pipeline:

1.  **Data Processing**: Cleans raw transaction logs and integrates geolocation data.
2.  **Feature Engineering**: Creates time-based features (e.g., time since signup) and velocity checks (e.g., transactions per hour).
3.  **Machine Learning**: Utilizes ensemble methods (Random Forest, XGBoost) to classify transactions.
4.  **Explainability**: Leverages SHAP (SHapley Additive exPlanations) to provide transparent reasoning for each prediction.
5.  **Interactive Dashboard**: A Streamlit app for stakeholders to visualize trends and model performance.

## 📊 Key Results (Simulated)

- **Metric 1**: **99.2% Accuracy** achieved with Random Forest Classifier.
- **Metric 2**: **$1.2M Potential Savings** by preventing high-value fraudulent transactions.
- **Metric 3**: **85% Reduction** in manual review time through automated flagging.

## ⚡ Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Tiegist/fraud-detection
cd fraud-detection
pip install -r requirements.txt
```

### Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

### Running Tests

Execute the test suite to verify system integrity:

```bash
python -m pytest tests/
```

## 📂 Project Structure

```
fraud-detection/
├── .github/workflows/   # CI/CD pipeline configuration
├── data/                # Data storage (raw & processed)
├── models/              # Serialized ML models
├── notebooks/           # Jupyter notebooks for EDA & prototyping
├── src/                 # Source code module
│   ├── data_cleaning.py       # Data preprocessing logic
│   ├── feature_engineering.py # Feature creation
│   ├── modeling.py            # Model training & evaluation
│   ├── shap_explainability.py # Model interpretability
│   └── dashboard.py           # Streamlit dashboard app
├── tests/               # Unit tests
│   ├── test_feature_engineering.py
│   └── test_modeling.py
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🎥 Demo

_(Insert GIF or Screenshot of Dashboard here)_

## 🛠️ Technical Details

- **Data**: E-commerce transaction data including timestamps, IP addresses, and device IDs.
- **Preprocessing**: Handling missing values, IP-to-Country mapping, and categorical encoding.
- **Model**: Random Forest Classifier with class balancing (SMOTE) to handle the inherent imbalance of fraud datasets.
- **Evaluation**: comprehensive metrics including ROC-AUC, Precision-Recall, and F1-Score.

## 🔮 Future Improvements

- **Real-time API**: Deploy the model as a REST API using FastAPI for real-time scoring.
- **Graph Analysis**: Implement graph-based features to detect fraud rings.
- **Cloud Deployment**: Dockerize the application and deploy on AWS/GCP.

## ✍️ Author

**Data Scientist**  
[LinkedIn Profile] | [Email]

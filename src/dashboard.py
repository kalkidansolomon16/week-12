import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaning import clean_fraud_data
from src.feature_engineering import create_time_features

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Fraud Detection Analytics and Explainability Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Fraud Analysis", "Model Explainability"])

# Load Data
@st.cache_data
def load_data():
    data_path = "data/raw/Fraud_Data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, nrows=50000) # Load subset for speed
        df = clean_fraud_data(df)
        df = create_time_features(df)
        return df
    return None

df = load_data()

if df is None:
    st.error("Data file not found. Please ensure 'data/raw/Fraud_Data.csv' exists.")
    st.stop()

if page == "Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{df['class'].sum():,}")
    col3.metric("Fraud Rate", f"{df['class'].mean()*100:.2f}%")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Data Distribution")
    fig = px.histogram(df, x="purchase_value", color="class", nbins=50, 
                       title="Purchase Value Distribution by Class",
                       labels={"class": "Is Fraud?"})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Fraud Analysis":
    st.header("🕵️ Fraud Patterns Analysis")
    
    if 'hour_of_day' in df.columns:
        st.subheader("Fraud by Hour of Day")
        fraud_by_hour = df.groupby('hour_of_day')['class'].mean().reset_index()
        fig = px.line(fraud_by_hour, x='hour_of_day', y='class', 
                      title="Fraud Rate by Hour of Day",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    if 'age' in df.columns:
        st.subheader("Fraud by Age")
        fig = px.box(df, x="class", y="age", title="Age Distribution for Fraud vs Non-Fraud")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Explainability":
    st.header("🧠 Model Explainability (SHAP)")
    st.info("This section demonstrates how SHAP values explain model predictions.")
    
    # Placeholder for actual model explanation
    st.subheader("Global Feature Importance (Simulation)")
    
    # Simulated Feature Importance
    features = ['purchase_value', 'time_since_signup', 'age', 'transaction_velocity']
    importance = [0.35, 0.25, 0.15, 0.10]
    
    fig = px.bar(x=importance, y=features, orientation='h', 
                 title="Top Factors Driving Fraud Predictions",
                 labels={'x': 'SHAP Importance', 'y': 'Feature'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Individual Prediction Explanation")
    st.markdown("""
    **Case ID: #12345**
    - **Prediction:** High Risk (85% probability)
    - **Top Contributors:**
        - 🔺 High Purchase Value ($5,000)
        - 🔺 Immediate Transaction after Signup
        - 🔻 Valid IP Location
    """)
    
    st.warning("To enable real-time explanations, train and save a model to the 'models/' directory.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & Plotly")

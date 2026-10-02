import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
# Load model
model = joblib.load(r"models\chrun_model.pkl")

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn risk.")

# Inputs
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

if st.button("Predict Churn"):

    data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "InternetService": [internet]
    })

    probability = model.predict_proba(data)[0][1]

    st.metric("Churn Probability", f"{probability * 100:.1f}%")

    if probability >= 0.70:
        st.error("🔴 High Churn Risk")
    elif probability >= 0.40:
        st.warning("🟡 Medium Churn Risk")
    else:
        st.success("🟢 Low Churn Risk")
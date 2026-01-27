import streamlit as st
import requests
import os

# When running locally
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict"  # default for local / EC2
)

st.title("Salary Prediction App")

years_exp = st.number_input(
    "Enter years of experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

if st.button("Predict Salary"):
    response = requests.post(API_URL, params={"yr_exp": years_exp})

    if response.status_code == 200:
        result = response.json()

        salary = result["predicted_salary"]

        st.success(f"Predicted Salary: {salary:,.2f}")
    else:
        st.error("Failed to get prediction from API")

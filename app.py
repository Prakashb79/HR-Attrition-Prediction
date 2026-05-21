
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.set_page_config(page_title="HR Attrition Prediction", layout="wide")

st.title("HR Attrition Prediction System")
st.write("Predict whether an employee is likely to leave the company")

st.sidebar.header("Employee Details")

# Input fields
Age = st.sidebar.number_input("Age", 18, 60, 25)
DailyRate = st.sidebar.number_input("Daily Rate", 100, 2000, 500)
DistanceFromHome = st.sidebar.number_input("Distance From Home", 1, 50, 5)
MonthlyIncome = st.sidebar.number_input("Monthly Income", 1000, 50000, 10000)
TotalWorkingYears = st.sidebar.number_input("Total Working Years", 0, 40, 5)
YearsAtCompany = st.sidebar.number_input("Years At Company", 0, 40, 3)
JobSatisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
EnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
WorkLifeBalance = st.sidebar.slider("Work Life Balance", 1, 4, 3)

# Default values for remaining columns
input_data = {}

for col in features:
    input_data[col] = 0

# Assign values
if 'Age' in input_data:
    input_data['Age'] = Age

if 'DailyRate' in input_data:
    input_data['DailyRate'] = DailyRate

if 'DistanceFromHome' in input_data:
    input_data['DistanceFromHome'] = DistanceFromHome

if 'MonthlyIncome' in input_data:
    input_data['MonthlyIncome'] = MonthlyIncome

if 'TotalWorkingYears' in input_data:
    input_data['TotalWorkingYears'] = TotalWorkingYears

if 'YearsAtCompany' in input_data:
    input_data['YearsAtCompany'] = YearsAtCompany

if 'JobSatisfaction' in input_data:
    input_data['JobSatisfaction'] = JobSatisfaction

if 'EnvironmentSatisfaction' in input_data:
    input_data['EnvironmentSatisfaction'] = EnvironmentSatisfaction

if 'WorkLifeBalance' in input_data:
    input_data['WorkLifeBalance'] = WorkLifeBalance

# Create dataframe
input_df = pd.DataFrame([input_data])

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("Predict Attrition"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Employee is likely to leave the company")
    else:
        st.success("Employee is likely to stay in the company")

    st.write(f"Attrition Probability: {probability:.2f}")

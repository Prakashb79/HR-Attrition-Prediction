import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("HR Attrition Prediction")

# User Inputs
Age = st.number_input("Age", 18, 60, 25)
DailyRate = st.number_input("Daily Rate", 100, 2000, 500)
DistanceFromHome = st.number_input("Distance From Home", 1, 50, 5)
Education = st.slider("Education", 1, 5, 3)
EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
HourlyRate = st.number_input("Hourly Rate", 10, 100, 50)
JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
JobLevel = st.slider("Job Level", 1, 5, 2)
JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 2)
MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 10000)
MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 10000)
NumCompaniesWorked = st.slider("Num Companies Worked", 0, 10, 2)
PercentSalaryHike = st.slider("Percent Salary Hike", 10, 30, 15)
PerformanceRating = st.slider("Performance Rating", 1, 4, 3)
RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
StockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)
TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 3)
WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 2)
YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
YearsInCurrentRole = st.slider("Years In Current Role", 0, 20, 3)
YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, 3)

# Categorical Inputs
BusinessTravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
Gender = st.selectbox("Gender", ["Male", "Female"])
JobRole = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative",
    "Manager", "Sales Representative", "Research Director", "Human Resources"
])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
OverTime = st.selectbox("Over Time", ["Yes", "No"])

# Create dataframe
input_data = pd.DataFrame({
    'Age': [Age],
    'BusinessTravel': [BusinessTravel],
    'DailyRate': [DailyRate],
    'Department': [Department],
    'DistanceFromHome': [DistanceFromHome],
    'Education': [Education],
    'EducationField': [EducationField],
    'EmployeeCount': [1],
    'EmployeeNumber': [1],
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'Gender': [Gender],
    'HourlyRate': [HourlyRate],
    'JobInvolvement': [JobInvolvement],
    'JobLevel': [JobLevel],
    'JobRole': [JobRole],
    'JobSatisfaction': [JobSatisfaction],
    'MaritalStatus': [MaritalStatus],
    'MonthlyIncome': [MonthlyIncome],
    'MonthlyRate': [MonthlyRate],
    'NumCompaniesWorked': [NumCompaniesWorked],
    'Over18': ['Y'],
    'OverTime': [OverTime],
    'PercentSalaryHike': [PercentSalaryHike],
    'PerformanceRating': [PerformanceRating],
    'RelationshipSatisfaction': [RelationshipSatisfaction],
    'StandardHours': [80],
    'StockOptionLevel': [StockOptionLevel],
    'TotalWorkingYears': [TotalWorkingYears],
    'TrainingTimesLastYear': [TrainingTimesLastYear],
    'WorkLifeBalance': [WorkLifeBalance],
    'YearsAtCompany': [YearsAtCompany],
    'YearsInCurrentRole': [YearsInCurrentRole],
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
    'YearsWithCurrManager': [YearsWithCurrManager]
})

# Prediction
# Prediction
if st.button("Predict Attrition"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Employee is likely to leave the company")
    else:
        st.success("Employee is likely to stay in the company")

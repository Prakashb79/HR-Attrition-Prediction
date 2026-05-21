import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("HR Attrition Prediction")

# Input
age = st.number_input("Age", 18, 60, 25)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age]
    })

    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")

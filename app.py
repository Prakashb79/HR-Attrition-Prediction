import streamlit as st

st.title("HR Attrition Prediction App")

age = st.number_input("Enter Age", 18, 60, 25)

if st.button("Predict"):
    st.success(f"App Working Successfully. Age entered: {age}")

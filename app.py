
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load files
model = pickle.load(open('model.pkl', 'rb'))
st.title("HR Attrition Prediction") 
# Example input age = st.number_input("Age", 18, 60, 25) 
if st.button("Predict"): 
    input_data = pd.DataFrame(
        { "Age": [age]
        }) 
    try: 
        prediction = model.predict(input_data) 
        st.success(f"Prediction: {prediction[0]}") 
    except Exception as e: st.error(e)

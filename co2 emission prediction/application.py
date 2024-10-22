import streamlit as st
import pandas as pd
import numpy as np
from model import predict_co2  # Import the function from model.py

# Streamlit UI Components
st.title("CO2 Emission Prediction")

st.write("""
### Enter the vehicle details below to predict the CO2 emission.
""")

# Input fields for user to enter data
car = st.text_input("Car & Model:", placeholder="e.g. Ford Fiesta")
volume = st.number_input("Engine Volume (cc):", min_value=0.0, format="%.2f")
weight = st.number_input("Weight (kg):", min_value=0.0, format="%.2f")

# Button to trigger prediction
if st.button("Predict CO2 Emission"):
    if volume > 0 and weight > 0:
        # Call the prediction function from the model
        co2_pred = predict_co2(volume, weight)
        st.success(f'Predicted CO2 Emission for {car}: {co2_pred:.2f} g/km')
    else:
        st.error("Please enter valid values for both engine volume and weight.")

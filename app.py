#streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Hivi Project Predictor", layout="centered")

st.title("📊 Real-state price prediction")
st.write("Enter the Details and get the estimate value of Property prices")

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- User input section ---
st.header("Input Features")

# Example: replace with your actual feature names from the notebook
feature_names = [
    "Age (1 to 30)", "Floor (0 to 50)", "Bedroom (1 to 6)", "Bathroom (1 to 6)", "Area Sqft (200 to 5000)"
]

user_input = []
for feature in feature_names:
    val = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(val)

# Convert input to array and scale
input_array = np.array([user_input])
scaled_input = scaler.transform(input_array)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    st.success(f"Predicted Property Price: {prediction[0]:.2f} Lakh")

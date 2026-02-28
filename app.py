
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Title Section
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app predicts whether a patient is diabetic or not using Logistic Regression.</p>", unsafe_allow_html=True)

st.divider()

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

st.divider()

if st.button("Predict", use_container_width=True):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk: The patient is likely Diabetic.")
    else:
        st.success("✅ Low Risk: The patient is likely Not Diabetic.")

st.divider()
st.caption("Developed by Aliya Afzal | BTech AI Project")

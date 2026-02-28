import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# Session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":

    st.markdown("""
        <h1 style='text-align: center; color: #ff4b4b;'>🩺 Diabetes Prediction System</h1>
        <h4 style='text-align: center;'>Early Detection Using Machine Learning</h4>
        <br>
    """, unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=200)

    st.markdown("""
    This application predicts whether a patient is likely to have diabetes  
    based on medical parameters using a Logistic Regression model.

    🔹 Model Accuracy: 77.6%  
    🔹 Algorithm: Logistic Regression  
    🔹 Deployment: Streamlit Cloud  
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start Prediction", use_container_width=True):
        st.session_state.page = "prediction"

# ---------------- PREDICTION PAGE ----------------
elif st.session_state.page == "prediction":

    st.title("Enter Patient Details")

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

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict", use_container_width=True):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠ High Risk: The patient is likely Diabetic.")
        else:
            st.success("✅ Low Risk: The patient is likely Not Diabetic.")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.caption("Developed by Aliya Afzal | BTech AI Project")

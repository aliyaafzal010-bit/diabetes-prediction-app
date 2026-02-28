import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Project", "Prediction"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Welcome to Diabetes Prediction System</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966489.png", width=150)
    
    st.write("""
    This web application predicts whether a patient is likely to have diabetes.
    
    🔹 Built using Machine Learning  
    🔹 Model Used: Logistic Regression  
    🔹 Deployed using Streamlit Cloud  
    """)
    
    st.info("Use the sidebar to navigate to the Prediction page.")

# ---------------- ABOUT PAGE ----------------
elif page == "About Project":
    st.title("About This Project")
    st.write("""
    This project was developed as part of a BTech AI/ML training project.

    📊 Dataset: PIMA Indian Diabetes Dataset  
    🤖 Model: Logistic Regression  
    🎯 Accuracy: 77.6%  
    ⚙ Deployment: Streamlit Cloud  

    The model predicts diabetes based on medical parameters such as glucose level, BMI, age, etc.
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
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

    if st.button("Predict", use_container_width=True):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠ High Risk: The patient is likely Diabetic.")
        else:
            st.success("✅ Low Risk: The patient is likely Not Diabetic.")

    st.caption("Developed by Aliya Afzal | BTech AI Project")

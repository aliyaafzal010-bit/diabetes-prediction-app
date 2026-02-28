import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ----------- PROFESSIONAL LIGHT CSS -----------
st.markdown("""
<style>

/* Main background white */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #1a1a1a;
}

/* Sidebar light grey */
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}

/* Force all normal text dark */
html, body, [class*="css"]  {
    color: #1a1a1a !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Headings */
h1 {
    color: #0f172a;
    font-weight: 700;
}

h2, h3 {
    color: #1e293b;
}

/* Button styling */
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: #1e40af;
    color: white;
}

/* Success & Error box */
.result-success {
    background-color: #e6f4ea;
    padding: 15px;
    border-radius: 8px;
    color: #166534;
    font-weight: 600;
}

.result-error {
    background-color: #fdecea;
    padding: 15px;
    border-radius: 8px;
    color: #991b1b;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ----------- SIDEBAR NAVIGATION -----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# ----------- HOME -----------
if page == "Home":
    st.title("🩺 Diabetes Prediction System")

    st.write("""
    ### Welcome 👋

    This web application uses a **Logistic Regression Machine Learning Model**
    to predict whether a person is likely to have diabetes.

    It analyzes important medical parameters such as:
    - Glucose Level
    - BMI
    - Blood Pressure
    - Insulin Level
    - Age

    👉 Go to **Prediction** from the sidebar to test the model.
    """)

# ----------- PREDICTION -----------
elif page == "Prediction":

    st.title("🔍 Enter Patient Details")

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

    if st.button("Predict Result"):

        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.markdown('<div class="result-error">The patient is likely Diabetic.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-success">The patient is likely Not Diabetic.</div>', unsafe_allow_html=True)

# ----------- ABOUT -----------
elif page == "About":

    st.title("📘 About This Project")

    st.write("""
    This project was developed as a **BTech AI/ML Mini Project**.

    ### Model Used:
    Logistic Regression

    ### Dataset:
    PIMA Indians Diabetes Dataset

    ### Objective:
    To build a machine learning model for early diabetes prediction.

    ### Developed By:
    Aliya Afzal
    """)

    st.markdown("---")
    st.caption("Diabetes Prediction System | 2026")

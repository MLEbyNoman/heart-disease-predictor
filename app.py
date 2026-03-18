
import streamlit as st
import numpy as np
import pickle

# -------------------------
# Load Model & Scaler
# -------------------------
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="GP Assistant", layout="centered")

st.title("🏥 Smart GP Assistant")
st.subheader("Heart Disease Risk Prediction")

# -------------------------
# Input Section
# -------------------------
st.markdown("### Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain (0-3)", [0,1,2,3])
    trestbps = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0,1])

with col2:
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("CA (vessels)", [0,1,2,3])
    thal = st.selectbox("Thal", [1,2,3])

# -------------------------
# Prediction Logic
# -------------------------
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    # Risk Levels
    if prob > 0.7:
        risk = "🔴 HIGH"
        color = "red"
    elif prob > 0.4:
        risk = "🟡 MEDIUM"
        color = "orange"
    else:
        risk = "🟢 LOW"
        color = "green"

    st.markdown("### Result")

    st.markdown(f"<h2 style='color:{color};'>Risk Level: {risk}</h2>", unsafe_allow_html=True)
    st.write(f"Confidence: {prob*100:.2f}%")

# -------------------------
# Disclaimer
# -------------------------
st.warning("⚠ This is not a medical diagnosis. Consult a doctor.")

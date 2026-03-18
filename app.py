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

# -------------------------
# Header (Improved UI)
# -------------------------
st.markdown(
"""
<h1 style='text-align: center; color: #2E86C1;'>🏥 Smart GP Assistant</h1>
<h3 style='text-align: center;'>Heart Disease Risk Prediction System</h3>
<hr>
""",
unsafe_allow_html=True
)

# -------------------------
# Input Section
# -------------------------
st.markdown("### 📝 Enter Patient Details")

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

    # -------------------------
    # Result Section (Improved)
    # -------------------------
    st.markdown("### 🩺 Prediction Result")

    if prob > 0.7:
        st.error(f"🔴 HIGH RISK ({prob*100:.2f}%)")
    elif prob > 0.4:
        st.warning(f"🟡 MEDIUM RISK ({prob*100:.2f}%)")
    else:
        st.success(f"🟢 LOW RISK ({prob*100:.2f}%)")

    # -------------------------
    # Explanation Section
    # -------------------------
    st.markdown("### 🧠 Explanation")

    if prob > 0.7:
        st.write("High risk detected due to abnormal heart indicators. Immediate medical consultation is recommended.")
    elif prob > 0.4:
        st.write("Moderate risk. Regular monitoring and lifestyle improvements are advised.")
    else:
        st.write("Low risk. Maintain a healthy lifestyle and regular checkups.")

    # -------------------------
    # Important Features Info
    # -------------------------
    st.markdown("### 📊 Key Risk Factors")

    st.write("""
    - Chest Pain Type  
    - Cholesterol Level  
    - Blood Pressure  
    - Maximum Heart Rate  
    - Exercise-induced Angina  
    """)

# -------------------------
# Footer / Disclaimer
# -------------------------
st.markdown("---")
st.warning("⚠ This tool is for educational purposes only. Always consult a qualified doctor.")

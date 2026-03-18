import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load Model & Scaler
# -------------------------
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Feature names
features = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="GP Assistant", layout="centered")

# -------------------------
# Header
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
# Prediction
# -------------------------
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    # -------------------------
    # Result
    # -------------------------
    st.markdown("### 🩺 Prediction Result")

    if prob > 0.7:
        st.error(f"🔴 HIGH RISK ({prob*100:.2f}%)")
    elif prob > 0.4:
        st.warning(f"🟡 MEDIUM RISK ({prob*100:.2f}%)")
    else:
        st.success(f"🟢 LOW RISK ({prob*100:.2f}%)")

    # -------------------------
    # Explanation
    # -------------------------
    st.markdown("### 🧠 Explanation")

    if prob > 0.7:
        st.write("High risk detected due to abnormal heart indicators.")
    elif prob > 0.4:
        st.write("Moderate risk. Monitor your health.")
    else:
        st.write("Low risk. Maintain healthy lifestyle.")

    # -------------------------
    # Feature Importance Graph
    # -------------------------
    st.markdown("### 📊 Feature Importance")

    try:
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.set_title("Feature Importance")

        st.pyplot(fig)

    except:
        st.warning("Feature importance not available for this model.")

    # -------------------------
    # Patient Input Visualization
    # -------------------------
    st.markdown("### 📈 Patient Data Overview")

    input_df = pd.DataFrame(input_data, columns=features).T
    input_df.columns = ["Value"]

    st.bar_chart(input_df)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.warning("⚠ This tool is for educational purposes only. Consult a doctor.")

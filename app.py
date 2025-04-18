import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("🫀 Heart Disease Prediction App")

# Load the trained model
model = joblib.load("model.pkl")

# Sidebar input fields
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (0, 1, 2, 3)", [0, 1, 2, 3])

# Convert gender
sex_val = 1 if sex == "Male" else 0

# Create input DataFrame that matches model training
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex_val,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}])

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

# Optional visualization
st.subheader("📊 Feature Correlation Heatmap")
try:
    df = pd.read_csv("heart_disease_dataset.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("Dataset not found. Please make sure 'heart_disease_dataset.csv' is in the project folder.")

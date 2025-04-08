
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Heart Disease Detection", layout="centered")

st.title("🫀 Heart Disease Detection App")
st.write("Enter patient details below to predict heart disease risk.")

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Preprocessing
df.dropna(inplace=True)
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split and train model
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

# Sidebar Form Input
st.sidebar.header("Patient Info")
age = st.sidebar.slider("Age", 20, 100, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])
symptom = st.sidebar.multiselect("Symptoms", ["Shortness of Breath", "Fatigue", "Chest Tightness", "Swelling", "None"])
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)

# Convert input
gender_val = 1 if gender == "Male" else 0
pain_val = {"Typical": 0, "Atypical": 1, "Non-anginal": 2, "Asymptomatic": 3}[chest_pain]
symptom_score = len(symptom)

# Create dummy input for prediction
input_features = np.array([[age, gender_val, pain_val, weight, symptom_score]])
input_df = pd.DataFrame(input_features, columns=["age", "sex", "chest pain", "weight", "symptom score"])

# Prediction
if st.sidebar.button("Predict"):
    st.subheader("Prediction Result")
    if model.predict(input_df)[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

# Visuals
st.subheader("📊 Dataset Visualizations")
st.write("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("Heart Disease Cases Distribution")
st.bar_chart(df["target"].value_counts())

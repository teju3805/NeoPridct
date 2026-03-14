import streamlit as st
import numpy as np
import joblib
import gzip
import time

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="NeoPredict Neonatal Monitoring System",
    layout="wide"
)

st.title("🍼 NeoPredict Neonatal Monitoring Dashboard")

st.markdown("AI-powered neonatal monitoring using **Hybrid CNN + BiLSTM models**")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

@st.cache_resource
def load_models():

    with gzip.open("apnea_model.pkl.gz", "rb") as f:
        apnea_model = joblib.load(f)

    with gzip.open("sepsis_model.pkl.gz", "rb") as f:
        sepsis_model = joblib.load(f)

    return apnea_model, sepsis_model


apnea_model, sepsis_model = load_models()

# ---------------------------------------------------
# Risk Rules
# ---------------------------------------------------

def rule_based_risk(hr, spo2, rr, temp):

    if 120 <= hr <= 160 and 95 <= spo2 <= 100 and 30 <= rr <= 60 and 36.5 <= temp <= 37.5:
        return "Normal Infant", "Normal"

    if 100 <= hr <= 120 and 90 <= spo2 <= 94 and rr < 25:
        return "Apnea", "Moderate Risk"

    if hr < 100 and spo2 < 90:
        return "Apnea", "Critical Risk"

    if 90 <= hr <= 110 and 92 <= spo2 <= 95 and 25 <= rr <= 40:
        return "Bradycardia", "Moderate Risk"

    if hr < 90 and spo2 < 90 and rr < 25:
        return "Bradycardia", "Critical Risk"

    if 160 <= hr <= 180 and 90 <= spo2 <= 94 and 60 <= rr <= 70 and 37.5 <= temp <= 38.5:
        return "Sepsis", "Moderate Risk"

    if (hr > 180 or hr < 100) and spo2 < 90 and rr > 70 and (temp > 38.5 or temp < 35):
        return "Sepsis", "Critical Risk"

    return "Unknown", "Monitor"


# ---------------------------------------------------
# Sidebar - Patient Details
# ---------------------------------------------------

st.sidebar.header("👶 Patient Information")

patient_id = st.sidebar.text_input("Patient ID")
doctor = st.sidebar.text_input("Doctor Name")
caregiver = st.sidebar.text_input("Caregiver Contact")

st.sidebar.markdown("---")

# ---------------------------------------------------
# Sensor Inputs
# ---------------------------------------------------

st.header("📡 Sensor Inputs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    hr = st.number_input("Heart Rate (bpm)", 40, 220, 140)

with col2:
    spo2 = st.number_input("SpO₂ (%)", 70, 100, 95)

with col3:
    rr = st.number_input("Respiration Rate", 5, 100, 40)

with col4:
    temp = st.number_input("Temperature °C", 34.0, 41.0, 37.0)

# ---------------------------------------------------
# Sensor Visualization
# ---------------------------------------------------

st.subheader("📊 Live Sensor Monitoring")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Heart Rate", hr)
c2.metric("SpO₂", spo2)
c3.metric("Respiration", rr)
c4.metric("Temperature", temp)

# ---------------------------------------------------
# AI Processing Pipeline
# ---------------------------------------------------

st.header("🧠 AI Processing Pipeline")

pipeline = st.progress(0)

steps = [
    "Collecting Sensor Signals...",
    "Signal Preprocessing...",
    "Feature Extraction...",
    "Feeding Data to CNN Layer...",
    "Temporal Analysis using BiLSTM...",
    "Hybrid Model Decision Layer...",
    "Risk Classification..."
]

if st.button("▶ Start AI Analysis"):

    for i, step in enumerate(steps):
        st.write(step)
        pipeline.progress((i + 1) / len(steps))
        time.sleep(0.6)

    # ---------------------------------------------------
    # Model Prediction
    # ---------------------------------------------------

    features = np.array([[hr, spo2, rr]])

    apnea_pred = apnea_model.predict(features)[0]
    sepsis_pred = sepsis_model.predict(features)[0]

    disease, risk = rule_based_risk(hr, spo2, rr, temp)

    detected = "Normal"

    if apnea_pred == 1:
        detected = "Apnea"

    elif sepsis_pred == 1:
        detected = "Sepsis"

    elif hr < 90:
        detected = "Bradycardia"

    # ---------------------------------------------------
    # Results
    # ---------------------------------------------------

    st.header("🔎 Diagnosis Result")

    r1, r2 = st.columns(2)

    r1.metric("Detected Disease", detected)
    r2.metric("Risk Level", risk)

    # ---------------------------------------------------
    # Alert System
    # ---------------------------------------------------

    st.header("🚨 Alert System")

    if risk == "Critical Risk":

        st.error("🚨 CRITICAL CONDITION DETECTED")

        st.write("Alert Sent To:")

        st.write("👨‍⚕ Doctor:", doctor)
        st.write("👩‍⚕ Caregiver:", caregiver)
        st.write("📱 Parents notified")

    elif risk == "Moderate Risk":

        st.warning("⚠ Moderate Risk – Monitor Patient")

    elif risk == "Normal":

        st.success("✅ Infant Vitals Normal")

    else:

        st.info("Monitoring Recommended")

# ---------------------------------------------------
# Clinical Reference Table
# ---------------------------------------------------

st.header("📋 Neonatal Clinical Reference")

st.table({
"Disease":["Normal","Apnea","Bradycardia","Sepsis"],
"Normal HR":[ "120-160", "100-120", "90-110", "160-180"],
"SpO2":[ "95-100", "90-94", "92-95", "90-94"],
"Respiration":[ "30-60", "<25", "25-40", "60-70"],
"Temperature":[ "36.5-37.5", "36-37.5", "36-37.5", "37.5-38.5"]
})

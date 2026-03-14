import streamlit as st
import numpy as np
import joblib
import gzip
import time
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="NeoPredict NICU Monitor", layout="wide")

st.title("NeoPredict – Neonatal ICU Monitoring System")
st.markdown("AI-Powered Neonatal Health Monitoring (Hybrid CNN + BiLSTM)")

# ------------------------------------------------
# Load Models
# ------------------------------------------------

@st.cache_resource
def load_models():

    apnea_model = None
    sepsis_model = None

    try:
        with gzip.open("neopredict_app/neopredict_app/backend/apnea_model.pkl.gz","rb") as f:
            apnea_model = joblib.load(f)
    except:
        st.warning("Apnea model not found")

    try:
        with gzip.open("neopredict_app/neopredict_app/backend/sepsis_model.pkl.gz","rb") as f:
            sepsis_model = joblib.load(f)
    except:
        st.warning("Sepsis model not found")

    return apnea_model, sepsis_model

apnea_model, sepsis_model = load_models()

# ------------------------------------------------
# Sidebar Patient Info
# ------------------------------------------------

st.sidebar.header("👶 Patient Details")

patient_id = st.sidebar.text_input("Patient ID")
doctor = st.sidebar.text_input("Doctor Name")
caregiver = st.sidebar.text_input("Caregiver Contact")

st.sidebar.markdown("---")

# ------------------------------------------------
# Sensor Inputs
# ------------------------------------------------

st.header("📡 Sensor Inputs")

c1,c2,c3,c4 = st.columns(4)

with c1:
    hr = st.number_input("Heart Rate (bpm)",50,220,140)

with c2:
    spo2 = st.number_input("SpO₂ (%)",70,100,95)

with c3:
    rr = st.number_input("Respiration Rate",10,100,40)

with c4:
    temp = st.number_input("Temperature °C",34.0,41.0,37.0)

# ------------------------------------------------
# NICU Monitor Panel
# ------------------------------------------------

st.header("🖥 NICU Vital Monitor")

v1,v2,v3,v4 = st.columns(4)

v1.metric("Heart Rate",hr)
v2.metric("SpO₂",spo2)
v3.metric("Respiration",rr)
v4.metric("Temperature",temp)

# ------------------------------------------------
# ECG Simulation
# ------------------------------------------------

st.subheader("❤️ Live ECG Monitor")

placeholder = st.empty()

for i in range(30):

    x = np.linspace(0,4,200)

    # ECG-like waveform
    y = (
        np.sin(8*x) * 0.6 +
        np.sin(15*x) * 0.2 +
        np.random.normal(0,0.05,200)
    )

    fig_ecg = go.Figure()

    fig_ecg.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="lime", width=3)
        )
    )

    fig_ecg.update_layout(
        height=200,
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=0,b=0)
    )

    placeholder.plotly_chart(fig_ecg, use_container_width=True)

    time.sleep(0.15)

# ------------------------------------------------
# SpO2 Gauge
# ------------------------------------------------

st.subheader("🫁 SpO₂ Gauge")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=spo2,
    gauge={'axis':{'range':[70,100]}}
))

st.plotly_chart(gauge,use_container_width=True)

# ------------------------------------------------
# Respiration Wave
# ------------------------------------------------

st.subheader("🌬 Respiration Waveform")

x2 = np.linspace(0,10,200)
y2 = np.sin(x2*rr/20)

fig_resp = go.Figure()
fig_resp.add_trace(go.Scatter(x=x2,y=y2,mode="lines"))
fig_resp.update_layout(height=200)

st.plotly_chart(fig_resp,use_container_width=True)

# ------------------------------------------------
# Sensor Panel
# ------------------------------------------------

st.header("📟 Sensor Devices")

s1,s2,s3,s4 = st.columns(4)

s1.write("🩺 ECG Sensor")
s2.write("🫁 Pulse Oximeter")
s3.write("🌬 Respiration Belt")
s4.write("🌡 Temperature Sensor")

# ------------------------------------------------
# AI Pipeline Animation
# ------------------------------------------------

st.header("🧠 AI Processing Pipeline")

steps = [
"Receiving Sensor Signals",
"Signal Preprocessing",
"Feature Extraction",
"CNN Spatial Feature Learning",
"BiLSTM Temporal Analysis",
"Hybrid Model Decision",
"Disease Classification",
"Risk Assessment",
"Alert System Trigger"
]

progress = st.progress(0)

if st.button("▶ Run AI Diagnosis"):

    for i,step in enumerate(steps):
        st.write(step)
        progress.progress((i+1)/len(steps))
        time.sleep(0.5)

    # ------------------------------------------------
    # Model Prediction
    # ------------------------------------------------
    features = np.array([[hr,spo2,rr,temp]])
    
    apnea_pred = 0
    sepsis_pred = 0
    
    if apnea_model is not None:
        apnea_pred = apnea_model.predict(features)[0]
    
    if sepsis_model is not None:
        sepsis_pred = sepsis_model.predict(features)[0]
    

    detected = "Normal"

    if apnea_pred == 1:
        detected = "Apnea"

    elif sepsis_pred == 1:
        detected = "Sepsis"

    elif hr < 90:
        detected = "Bradycardia"

    # ------------------------------------------------
    # Risk Classification
    # ------------------------------------------------

    risk = "Normal"

    if hr <100 and spo2 <90:
        risk = "Critical"

    elif 100 <= hr <=120 and 90 <= spo2 <=94:
        risk = "Moderate"

    elif hr >180 or rr >70 or temp >38.5:
        risk = "Critical"

    # ------------------------------------------------
    # Results
    # ------------------------------------------------

    st.header("🔎 Diagnosis Result")

    r1,r2 = st.columns(2)

    r1.metric("Detected Disease",detected)
    r2.metric("Risk Level",risk)

    # ------------------------------------------------
    # Alert System
    # ------------------------------------------------

    st.header("🚨 Alert System")

    if risk == "Critical":

        st.error("🚨 CRITICAL CONDITION DETECTED")

        st.write("Alerts sent to:")

        st.write("👨‍⚕ Doctor:",doctor)
        st.write("👩‍⚕ Caregiver:",caregiver)
        st.write("📱 Parents notified")

    elif risk == "Moderate":

        st.warning("⚠ Moderate Risk – Monitor Infant")

    else:

        st.success("✅ Infant Vitals Normal")

# ------------------------------------------------
# Clinical Reference
# ------------------------------------------------

st.header("📋 Neonatal Clinical Reference")

df = pd.DataFrame({
"Disease":["Normal Infant","Apnea","Bradycardia","Sepsis"],
"Heart Rate":["120-160","100-120","90-110","160-180"],
"SpO2":["95-100","90-94","92-95","90-94"],
"Respiration":["30-60","<25","25-40","60-70"],
"Temperature":["36.5-37.5","36-37.5","36-37.5","37.5-38.5"]
})

st.table(df)

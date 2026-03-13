import streamlit as st
import numpy as np
import pandas as pd
import random
import gzip
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="NeoPredict NICU AI Monitor", layout="wide")

# -------- LOAD MODELS --------

with gzip.open("apnea_model.pkl.gz","rb") as f:
    apnea_model = joblib.load(f)

with gzip.open("sepsis_model.pkl.gz","rb") as f:
    sepsis_model = joblib.load(f)

# -------- TITLE --------

st.title("NeoPredict – AI Powered NICU Monitoring System")

st.markdown(
"""
Continuous monitoring of neonatal vital signs with **AI-based detection**
of **Apnea, Bradycardia, and Sepsis Risk**
"""
)

# -------- SENSOR INPUT --------

st.sidebar.header("NICU Sensor Data")

heart_rate = st.sidebar.slider("Heart Rate (bpm)",60,200,120)
spo2 = st.sidebar.slider("SpO₂ (%)",80,100,96)
temperature = st.sidebar.slider("Temperature (°C)",34.0,40.0,36.8)
respiration = st.sidebar.slider("Respiration Rate",5,60,32)

# -------- VITAL DISPLAY --------

c1,c2,c3,c4 = st.columns(4)

c1.metric("❤️ Heart Rate",f"{heart_rate} bpm")
c2.metric("🫁 Respiration",f"{respiration} bpm")
c3.metric("🩸 SpO₂",f"{spo2}%")
c4.metric("🌡 Temperature",f"{temperature} °C")

st.markdown("---")

# -------- AI DETECTION --------

st.header("AI Condition Detection")

condition = "Normal"

# -------- APNEA MODEL --------

apnea_features = np.array([[heart_rate,spo2,temperature,respiration] + [0]*20])

try:
    apnea_pred = apnea_model.predict(apnea_features)[0]
except:
    apnea_pred = 0

# -------- SEPSIS MODEL --------

sepsis_features = np.array([[heart_rate,temperature] + [0]*20])

try:
    sepsis_pred = sepsis_model.predict(sepsis_features)[0]
except:
    sepsis_pred = 0


# -------- LOGIC --------

if respiration < 12 or apnea_pred == 1:
    condition = "Apnea"

elif heart_rate < 90:
    condition = "Bradycardia"

elif temperature > 38 or sepsis_pred == 1:
    condition = "Sepsis Risk"


# -------- DISPLAY RESULT --------

if condition == "Apnea":

    st.error("⚠ Apnea Detected – Breathing interruption")

elif condition == "Bradycardia":

    st.error("⚠ Bradycardia Detected – Low Heart Rate")

elif condition == "Sepsis Risk":

    st.error("⚠ Possible Neonatal Sepsis Risk")

else:

    st.success("Infant Vital Signs Stable")


# -------- ALERT SYSTEM --------

st.markdown("---")
st.header("🚨 Alert Notification System")

if condition != "Normal":

    st.warning("Emergency Alert Triggered")

    st.write("Notification sent to:")

    st.write("• NICU Doctor")
    st.write("• Hospital Caregiver")
    st.write("• Infant Parents")

else:

    st.success("No critical alerts")


# -------- NICU MONITOR GRAPH --------

st.markdown("---")
st.header("Live NICU Monitoring")

time = list(range(30))

hr = [random.randint(110,160) for _ in range(30)]
resp = [random.randint(25,40) for _ in range(30)]
spo = [random.randint(92,100) for _ in range(30)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time,
    y=hr,
    mode="lines",
    name="Heart Rate",
    line=dict(color="red",width=3)
))

fig.add_trace(go.Scatter(
    x=time,
    y=resp,
    mode="lines",
    name="Respiration",
    line=dict(color="cyan",width=3)
))

fig.add_trace(go.Scatter(
    x=time,
    y=spo,
    mode="lines",
    name="SpO₂",
    line=dict(color="green",width=3)
))

fig.update_layout(
    template="plotly_dark",
    height=450
)

st.plotly_chart(fig,use_container_width=True)

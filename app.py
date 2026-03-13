import streamlit as st
import numpy as np
import pandas as pd
import random
import gzip
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="NeoPredict NICU Monitor", layout="wide")

# Load model
with gzip.open("apnea_model.pkl.gz","rb") as f:
    model = joblib.load(f)

st.title("NeoPredict – NICU Neonatal Monitoring System")

st.markdown("Real-time monitoring of neonatal vital signs with AI risk detection.")

# Sidebar (simulated sensors)
st.sidebar.header("Simulated Sensor Data")

heart_rate = st.sidebar.slider("Heart Rate (bpm)",80,180,120)
spo2 = st.sidebar.slider("SpO₂ (%)",80,100,96)
temperature = st.sidebar.slider("Temperature (°C)",35.0,40.0,36.8)
respiration = st.sidebar.slider("Respiration Rate (breaths/min)",20,60,32)

# NICU Vitals display
col1,col2,col3,col4 = st.columns(4)

col1.metric("❤️ Heart Rate",f"{heart_rate} bpm")
col2.metric("🩸 SpO₂",f"{spo2} %")
col3.metric("🌡 Temperature",f"{temperature} °C")
col4.metric("🫁 Respiration",f"{respiration} bpm")

st.markdown("---")

# AI Prediction
if st.button("Run AI Prediction"):

    features = np.array([[heart_rate,spo2,temperature,respiration] + [0]*20])

    # Simulated AI prediction logic

if heart_rate > 160 or spo2 < 90 or respiration > 50 or temperature > 38:

    st.error("⚠ Critical Risk Detected – Possible Neonatal Distress")
    st.write("Alert sent to doctors and parents")

elif heart_rate > 140 or respiration > 40:

    st.warning("⚠ Moderate Risk – Infant needs monitoring")

else:

    st.success("Infant Condition Stable")

    st.subheader("AI Risk Assessment")

    if prediction == 1:
        st.error("⚠ Critical Risk Detected — Possible Apnea")
        st.write("Alert sent to parents and doctors")
    else:
        st.success("Infant Condition Stable")

st.markdown("---")

# Heart rate monitoring graph
st.subheader("Live Heart Rate Monitoring")

time = list(range(30))
hr = [random.randint(110,150) for _ in range(30)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time,
    y=hr,
    mode="lines",
    line=dict(color="red",width=3)
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Heart Rate (bpm)",
    template="plotly_dark"
)

st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# Alert system
st.subheader("Alert System")

if spo2 < 90:
    st.error("⚠ Oxygen level critically low")
elif heart_rate > 160:
    st.warning("⚠ Abnormal heart rate detected")
elif respiration > 50:
    st.warning("⚠ Abnormal respiration detected")
else:
    st.success("No critical alerts")

import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go

st.set_page_config(page_title="NeoPredict NICU Monitor", layout="wide")

# ---------------- TITLE ----------------

st.title("NeoPredict – NICU Neonatal Monitoring System")
st.markdown("AI-powered monitoring for detecting **Apnea, Bradycardia, and Sepsis risk**")

# ---------------- SENSOR INPUT ----------------

st.sidebar.header("NICU Sensor Data")

heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 200, 120)
respiration = st.sidebar.slider("Respiration Rate (breaths/min)", 5, 60, 30)
spo2 = st.sidebar.slider("SpO₂ (%)", 80, 100, 96)
temperature = st.sidebar.slider("Temperature (°C)", 34.0, 40.0, 36.8)

# ---------------- VITAL DISPLAY ----------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("❤️ Heart Rate", f"{heart_rate} bpm")
col2.metric("🫁 Respiration", f"{respiration} bpm")
col3.metric("🩸 SpO₂", f"{spo2}%")
col4.metric("🌡 Temperature", f"{temperature} °C")

st.markdown("---")

# ---------------- CONDITION DETECTION ----------------

st.header("AI Condition Detection")

condition = "Normal"

if respiration < 12:
    condition = "Apnea"

elif heart_rate < 90:
    condition = "Bradycardia"

elif temperature > 38 and heart_rate > 150:
    condition = "Sepsis"

# ---------------- RESULT DISPLAY ----------------

if condition == "Apnea":
    st.error("⚠ Apnea Detected – Breathing interruption")

elif condition == "Bradycardia":
    st.error("⚠ Bradycardia Detected – Heart rate dangerously low")

elif condition == "Sepsis":
    st.error("⚠ Possible Neonatal Sepsis Detected")

else:
    st.success("Infant Vital Signs Stable")

st.markdown("---")

# ---------------- ALERT SYSTEM ----------------

st.header("🚨 Alert Notification System")

if condition != "Normal":

    st.warning("Emergency Alert Triggered")

    st.write("📩 Alert sent to:")

    st.write("• NICU Doctor")
    st.write("• Hospital Caregiver")
    st.write("• Infant Parents")

else:

    st.success("No critical alerts")

st.markdown("---")

# ---------------- LIVE MONITOR ----------------

st.header("Live NICU Vital Monitoring")

time = list(range(30))

heart_data = [random.randint(110,160) for _ in range(30)]
resp_data = [random.randint(25,40) for _ in range(30)]
spo_data = [random.randint(92,100) for _ in range(30)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time,
    y=heart_data,
    mode="lines",
    name="Heart Rate",
    line=dict(color="red", width=3)
))

fig.add_trace(go.Scatter(
    x=time,
    y=resp_data,
    mode="lines",
    name="Respiration",
    line=dict(color="cyan", width=3)
))

fig.add_trace(go.Scatter(
    x=time,
    y=spo_data,
    mode="lines",
    name="SpO₂",
    line=dict(color="green", width=3)
))

fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="Time",
    yaxis_title="Vital Signals"
)

st.plotly_chart(fig, use_container_width=True)

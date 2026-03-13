import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random
import plotly.express as px

st.set_page_config(page_title="NeoPredict", layout="wide")

st.title("NeoPredict – Neonatal Monitoring System")

# Load your trained model
import gzip
import joblib

with gzip.open("apnea_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

st.sidebar.header("Simulated Sensor Data")

heart_rate = st.sidebar.slider("Heart Rate (bpm)", 80,180,120)
spo2 = st.sidebar.slider("SpO2 (%)",80,100,96)
temp = st.sidebar.slider("Temperature (°C)",35.0,40.0,36.8)

# Show vitals
col1,col2,col3 = st.columns(3)

col1.metric("Heart Rate",f"{heart_rate} bpm")
col2.metric("SpO2",f"{spo2}%")
col3.metric("Temperature",f"{temp} °C")

st.markdown("---")

# Run AI prediction
if st.button("Run AI Prediction"):

    features = np.array([[heart_rate,spo2,temp]])

    prediction = model.predict(features)

    st.subheader("AI Prediction Result")

    if prediction == 1:
        st.error("⚠ Possible Apnea Detected")
        st.write("Alert sent to doctor and parents")
    else:
        st.success("Infant Condition Stable")

st.markdown("---")

# Monitoring graph
st.subheader("Heart Rate Trend")

data = pd.DataFrame({
"time":range(30),
"heart_rate":[random.randint(110,150) for _ in range(30)]
})

fig = px.line(data,x="time",y="heart_rate")

st.plotly_chart(fig,use_container_width=True)

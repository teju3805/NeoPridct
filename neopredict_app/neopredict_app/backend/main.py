from fastapi import FastAPI
import numpy as np
import joblib
import gzip

app = FastAPI()

# load models
with gzip.open("apnea_model.pkl.gz", "rb") as f:
    apnea_model = joblib.load(f)

with gzip.open("sepsis_model.pkl.gz", "rb") as f:
    sepsis_model = joblib.load(f)


@app.get("/")
def home():
    return {"message": "Neonatal Prediction API Running"}


@app.get("/predict_apnea")
def predict_apnea(hr: float, spo2: float, rr: float):

    # create 11 features because model expects 11
    data = np.zeros((1, 11))
    data[0][0] = hr
    data[0][1] = spo2
    data[0][2] = rr

    prediction = apnea_model.predict(data)

    return {"prediction": int(prediction[0])}


@app.get("/predict_sepsis")
def predict_sepsis(hr: float, spo2: float, rr: float):

    # create 42 features because model expects 42
    data = np.zeros((1, 42))
    data[0][0] = hr
    data[0][1] = spo2
    data[0][2] = rr

    prediction = sepsis_model.predict(data)

    return {"prediction": int(prediction[0])}

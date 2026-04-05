from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("../models/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Cancer Prediction API is running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}

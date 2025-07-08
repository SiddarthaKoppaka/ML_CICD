from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/model_v1.pkl")  # or load from registry/S3


@app.get("/")
def read_root():
    return {"message": "ML Model Inference API is live!"}


@app.post("/predict/")
def predict(input: dict):
    df = pd.DataFrame([input])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

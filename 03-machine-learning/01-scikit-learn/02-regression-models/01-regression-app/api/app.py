from fastapi import FastAPI
import joblib
from src.predict import predict_price

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/")
def home():
    return {"message": "Regression API Running"}

@app.post("/predict")
def predict(data: list):
    result = predict_price(model, scaler, data)
    return {"predicted_price": result}

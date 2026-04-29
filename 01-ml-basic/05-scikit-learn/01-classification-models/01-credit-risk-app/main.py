import pandas as pd
from src.data import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.utils import evaluate
import joblib


# Load data
df = load_data("data/loan_data.csv")

# Preprocess
x, y, scaler = preprocess_data(df)

# Train model
model, x_test, y_test = train_model(x, y)

# Evaluate
evaluate(model, x_test, y_test)

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully")

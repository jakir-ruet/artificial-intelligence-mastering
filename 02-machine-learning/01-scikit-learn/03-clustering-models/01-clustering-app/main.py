import joblib
from src.data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate

# Load data
df = load_data("data/customer_data.csv")

# Preprocess
x, scaler = preprocess(df)

# Train model
model = train_model(x)

# Evaluate
evaluate(model, x)

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully")

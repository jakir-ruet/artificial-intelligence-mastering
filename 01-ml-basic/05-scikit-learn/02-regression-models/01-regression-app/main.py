import joblib
from src.data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate

# Load data
df = load_data("data/house_data.csv")

# Preprocess
x, y, scaler = preprocess(df)

# Train
model, x_test, y_test = train_model(x, y)

# Evaluate
evaluate(model, x_test, y_test)

# Save model
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully")

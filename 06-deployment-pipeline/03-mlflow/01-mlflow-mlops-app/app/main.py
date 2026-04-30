from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load latest model from MLFlow
model = mlflow.pyfunc.load_model("/model")

@app.post("/predict")

def predict(area: float, bedrooms: float, age: float):
	input_data = np.array([[area, bedrooms, age]])
	prediction = model.predict(input_data)

	return {
		"predicted_price": float(prediction[0])
	}

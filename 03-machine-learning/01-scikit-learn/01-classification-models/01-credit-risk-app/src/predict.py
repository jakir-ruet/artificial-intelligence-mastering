import numpy as pd

def predict_risk(model, scaler, input_data):
	input_scaled = scaler.transform([input_data])
	prediction = model.predict(input_scaled)

	return "High Risk " if prediction[0] == 1 else "Low Risk"

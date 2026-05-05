import numpy as np

def predict_price(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)

    return prediction[0]

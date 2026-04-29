def predict_cluster(model, scaler, input_data):
    scaled = scaler.transform([input_data])
    cluster = model.predict(scaled)

    return cluster[0]

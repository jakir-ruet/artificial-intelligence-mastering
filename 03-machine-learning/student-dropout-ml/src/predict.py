import joblib
import pandas as pd

# MODEL_PATH = "models/dropout_model.pkl"
# Model update
MODEL_PATH = "models/best_dropout_model.pkl"
SCALER_PATH = "models/scaler.pkl"


FEATURE_COLUMNS = [
    "attendance_rate",
    "avg_marks",
    "failed_subjects",
    "fee_delay_days",
    "guardian_income",
    "disciplinary_actions",
    "previous_dropout_risk",
]


def predict_dropout(student_data):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame([student_data], columns=FEATURE_COLUMNS)
    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return {
        "dropout_prediction": "YES" if prediction == 1 else "NO",
        "risk_probability": round(float(probability), 4),
        "risk_level": get_risk_level(probability),
    }


def get_risk_level(probability):
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    return "LOW"


if __name__ == "__main__":
    student = {
        "attendance_rate": 58,
        "avg_marks": 42,
        "failed_subjects": 3,
        "fee_delay_days": 45,
        "guardian_income": 20000,
        "disciplinary_actions": 2,
        "previous_dropout_risk": 1,
    }

    result = predict_dropout(student)
    print(result)

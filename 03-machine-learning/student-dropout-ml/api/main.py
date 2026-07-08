import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from predict import predict_dropout

app = FastAPI(title="Student Dropout Prediction API")


class StudentRequest(BaseModel):
    attendance_rate: float
    avg_marks: float
    failed_subjects: int
    fee_delay_days: int
    guardian_income: float
    disciplinary_actions: int
    previous_dropout_risk: int


@app.get("/")
def health_check():
    return {"status": "UP", "service": "Student Dropout Prediction API"}


@app.post("/api/predict-dropout")
def predict(request: StudentRequest):
    student_data = request.model_dump()
    return predict_dropout(student_data)

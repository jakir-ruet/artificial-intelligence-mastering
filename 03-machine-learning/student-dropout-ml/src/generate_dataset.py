import numpy as np
import pandas as pd

# Reproducible results
np.random.seed(42)

TOTAL_STUDENTS = 1000

# Generate features
student_id = np.arange(1, TOTAL_STUDENTS + 1)

attendance_rate = np.random.uniform(40, 100, TOTAL_STUDENTS).round(2)

avg_marks = np.random.uniform(30, 95, TOTAL_STUDENTS).round(2)

failed_subjects = np.random.randint(0, 6, TOTAL_STUDENTS)

fee_delay_days = np.random.randint(0, 91, TOTAL_STUDENTS)

guardian_income = np.random.randint(15000, 100001, TOTAL_STUDENTS)

disciplinary_actions = np.random.randint(0, 5, TOTAL_STUDENTS)

previous_dropout_risk = np.random.randint(0, 2, TOTAL_STUDENTS)

# Calculate synthetic dropout risk score
risk_score = (
    (100 - attendance_rate) * 0.30
    + (100 - avg_marks) * 0.25
    + failed_subjects * 5
    + fee_delay_days * 0.10
    + disciplinary_actions * 3
    + previous_dropout_risk * 15
)

# Convert risk score into label
dropout = (risk_score >= 45).astype(int)

# Create DataFrame
df = pd.DataFrame(
    {
        "student_id": student_id,
        "attendance_rate": attendance_rate,
        "avg_marks": avg_marks,
        "failed_subjects": failed_subjects,
        "fee_delay_days": fee_delay_days,
        "guardian_income": guardian_income,
        "disciplinary_actions": disciplinary_actions,
        "previous_dropout_risk": previous_dropout_risk,
        "dropout": dropout,
    }
)

# Save dataset
df.to_csv("data/student_dropout_dataset.csv", index=False)

print("Dataset generated successfully.")
print()
print(df.head())

print()
print("Dataset shape:")
print(df.shape)

print()
print("Dropout distribution:")
print(df["dropout"].value_counts())

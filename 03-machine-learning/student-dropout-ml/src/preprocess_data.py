import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/student_dropout_dataset.csv"
SCALER_PATH = "models/scaler.pkl"


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def clean_dataset(df):
    df = df.drop_duplicates()

    df = df[
        (df["attendance_rate"] >= 0)
        & (df["attendance_rate"] <= 100)
        & (df["avg_marks"] >= 0)
        & (df["avg_marks"] <= 100)
        & (df["failed_subjects"] >= 0)
        & (df["fee_delay_days"] >= 0)
        & (df["guardian_income"] >= 0)
        & (df["disciplinary_actions"] >= 0)
    ]

    df = df.fillna(df.median(numeric_only=True))

    return df


def prepare_features(df):
    feature_columns = [
        "attendance_rate",
        "avg_marks",
        "failed_subjects",
        "fee_delay_days",
        "guardian_income",
        "disciplinary_actions",
        "previous_dropout_risk",
    ]

    X = df[feature_columns]
    y = df["dropout"]

    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess():
    df = load_dataset()
    df = clean_dataset(df)
    X, y = prepare_features(df)
    return split_and_scale(X, y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()

    print("Preprocessing completed.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

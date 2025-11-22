# demo_heart_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport

# -----------------------------------------------------------------------------
# STEP 1: Load the Dataset
# -----------------------------------------------------------------------------

df = pd.read_csv("./heart.csv")
print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------------------------------------------------------
# STEP 2: Check and Handle Missing Values (if any)
# -----------------------------------------------------------------------------

print("\nMissing values:\n", df.isnull().sum())

# -----------------------------------------------------------------------------
# STEP 3: Encode a Binary Categorical Feature
# -----------------------------------------------------------------------------

df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})  # Temporarily make it categorical
df['Sex_encoded'] = df['Sex'].map({'Female': 0, 'Male': 1})  # Re-encode manually

print("\nEncoded 'Sex' column preview:\n", df[['Sex', 'Sex_encoded']].head())

# -----------------------------------------------------------------------------
# STEP 4: Feature Creation - BMI-like Feature
# -----------------------------------------------------------------------------

df['chol_age_ratio'] = df['Cholesterol'] / (df['Age'] + 1)  # avoid division by zero

print("\nNew feature 'chol_age_ratio' preview:\n", df[['Cholesterol', 'Age', 'chol_age_ratio']].head())

# -----------------------------------------------------------------------------
# STEP 5: Normalize Numerical Features
# -----------------------------------------------------------------------------

scaler = StandardScaler()
df[['RestingBP_scaled', 'chol_age_ratio_scaled']] = scaler.fit_transform(df[['RestingBP', 'chol_age_ratio']])

print("\nScaled features preview:\n", df[['RestingBP', 'RestingBP_scaled', 'chol_age_ratio_scaled']].head())

# -----------------------------------------------------------------------------
# STEP 6: Validate with Profiling
# -----------------------------------------------------------------------------

profile = ProfileReport(df, title="Heart Disease Feature Engineering Report", explorative=True)
profile.to_file("./index.html")

print("\n✅ Profiling report saved: 'index.html'")

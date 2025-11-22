import os
import time
import joblib
import psutil
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# 1. Reload data
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None,
    names=[
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
)

# Clean and preprocess just like before
df.replace(" ?", np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop("income", axis=1)
y = df["income"].apply(lambda x: x.strip() == ">50K")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 2. Load previously saved model
model_file = "MLPClassifier.joblib"  # or use "LogisticRegression.joblib"
if not os.path.exists(model_file):
    print(f"Error: {model_file} not found. Run training script first.")
    exit(1)

pipeline = joblib.load(model_file)

# 3. Define test sample (single row, same structure)
sample = X_test.iloc[[0]]  # preserves DataFrame shape

# 4. Print available CPUs
cpu_list = psutil.cpu_count(logical=True)
print(f"\nAvailable logical CPUs: {cpu_list}")
print("Pinning to a single core for stress test...\n")

# 5. Pin to a single CPU core
p = psutil.Process()
original_affinity = p.cpu_affinity()
p.cpu_affinity([0])  # Pin to CPU core 0

# 6. Warmup + timing
print("Running 1000 inferences on single CPU core...\n")
start_time = time.perf_counter()
for _ in range(1000):
    _ = pipeline.predict(sample)
elapsed = time.perf_counter() - start_time
latency_ms = (elapsed / 1000) * 1000

# 7. Restore original CPU affinity
p.cpu_affinity(original_affinity)

# 8. Output results
print("===== Latency Results =====")
print(f"Model: {model_file}")
print(f"Total time for 1000 inferences (1 core): {elapsed:.3f} seconds")
print(f"Avg latency per inference: {latency_ms:.3f} ms")
print("===========================\n")


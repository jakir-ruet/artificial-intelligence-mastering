# 1. Import Libraries
import pandas as pd
from ydata_profiling import ProfileReport

# 2. Load Dataset
df = pd.read_csv("./heart-cleveland-upload.csv")

# 3. Preview Data
print("=== HEAD ===")
print(df.head())
print("\n=== INFO ===")
print(df.info())
print("\n=== SUMMARY STATISTICS ===")
print(df.describe())

# 4. Check for Missing Values and Duplicates
print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

# 5. Handle Duplicates and Missing Values
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

# 6. Data Type Conversion for Label Column (if needed)
# Change 'target' to 'condition' based on actual dataset
if df['condition'].dtype != 'int64':
    df['condition'] = df['condition'].astype(int)

# 7. Generate Data Profile Report
profile = ProfileReport(df, title="Heart Disease Data Profile", explorative=True)
profile.to_file("heart_data_profile.html")

# 8. Export Cleaned Dataset
df.to_csv("validated_heart_data.csv", index=False)
df.to_json("validated_heart_data.json", orient="records", lines=True)

print("✅ Data validation complete. Files exported.")

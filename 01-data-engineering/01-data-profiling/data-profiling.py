import pandas as pd
from ydata_profiling import ProfileReport

# Load sample CSV data
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

# Basic inspection
print(df.head())

# Full data profiling
profile = ProfileReport(df, title="Data Profile Report", explorative=True)
profile.to_file("./index.html")

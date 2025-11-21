# Import pandas library
import pandas as pd

# Step 1: Load the CSV file into a DataFrame
csv_file = './hris.csv'  # Ensure this file is in your working directory
df = pd.read_csv(csv_file)

# Step 2: Display the first few rows of the DataFrame
print(df.head())

# Step 3: Display basic information about the DataFrame
print(df.info())

# Step 4: Display summary statistics
print(df.describe())

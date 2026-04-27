def analyze_data(df):
    result = df.groupby('department')['salary'].mean()
    print("Average Salary by Department:\n", result)

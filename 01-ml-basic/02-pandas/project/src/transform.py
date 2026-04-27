import pandas as pd

def transform_data(df):
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Create new column
    df['salary_after_tax'] = df['salary'] * 0.9

    return df

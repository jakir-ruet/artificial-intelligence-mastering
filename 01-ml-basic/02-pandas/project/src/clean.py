def clean_data(df):
    # Fill missing age with median
    df['age'] = df['age'].fillna(df['age'].median())

    # Fill missing salary with mean
    df['salary'] = df['salary'].fillna(df['salary'].mean())

    # Drop rows where date is missing
    df = df.dropna(subset=['date'])

    return df

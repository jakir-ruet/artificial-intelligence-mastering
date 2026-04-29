from sklearn.preprocessing import StandardScaler

def preprocess(df):
    x = df.drop("customer_id", axis=1)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return x_scaled, scaler

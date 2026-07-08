from sklearn.preprocessing import StandardScaler

def preprocess(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    return x_scaled, y, scaler

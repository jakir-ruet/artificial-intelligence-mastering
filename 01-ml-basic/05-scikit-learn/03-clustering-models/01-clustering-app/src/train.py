from sklearn.cluster import KMeans

def train_model(x):
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(x)

    print("Model trained")
    return model

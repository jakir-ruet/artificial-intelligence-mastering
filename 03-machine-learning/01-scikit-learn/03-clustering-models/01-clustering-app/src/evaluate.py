from sklearn.metrics import silhouette_score

def evaluate(model, x):
    labels = model.labels_

    score = silhouette_score(x, labels)

    print("Silhouette Score:", score)

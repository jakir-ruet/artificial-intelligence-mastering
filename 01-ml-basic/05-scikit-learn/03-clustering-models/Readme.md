#### Clustering Evaluation Metrics

| Metric                    | Meaning                                                                           | When to Use                           |
| ------------------------- | --------------------------------------------------------------------------------- | ------------------------------------- |
| Silhouette Score          | Measures how well a data point fits within its cluster compared to other clusters | General clustering quality evaluation |
| Davies-Bouldin Index      | Measures similarity between clusters (lower value = better separation)            | Comparing clustering models           |
| Calinski-Harabasz Index   | Ratio of between-cluster variance to within-cluster variance                      | Evaluating cluster separation quality |
| Inertia                   | Sum of squared distances between samples and cluster centroids                    | K-Means optimization (Elbow method)   |
| Adjusted Rand Index (ARI) | Measures similarity between predicted clusters and true labels                    | When ground truth labels exist        |

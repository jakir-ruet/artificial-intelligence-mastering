### Clustering

It's a type of unsupervised learning where the goal is to group similar data points together without using labeled outputs.

> In simple words: Clustering = finding hidden groups in data automatically

**Example**

| Cluster | Use Case              | Features Used                  | Customer Type          | Characteristics                                  |
| ------- | --------------------- | ------------------------------ | ---------------------- | ------------------------------------------------ |
| 1       | Customer Segmentation | Income, Spending Behavior, Age | High Value Customers   | High income, high spending, premium buyers       |
| 2       | Customer Segmentation | Income, Spending Behavior, Age | Low Value Customers    | Low income, low spending, budget-conscious users |
| 3       | Customer Segmentation | Income, Spending Behavior, Age | Medium Value Customers | Medium income, moderate spending behavior        |

#### Types of Clustering

| Type               | Description                                   | Example                      |
| ------------------ | --------------------------------------------- | ---------------------------- |
| Partition-based    | Divides data into K clusters                  | K-Means                      |
| Hierarchical       | Builds tree-like cluster structure            | Agglomerative Clustering     |
| Density-based      | Groups based on dense regions                 | DBSCAN                       |
| Distribution-based | Assumes data follows probability distribution | Gaussian Mixture Model (GMM) |
| Grid-based         | Divides space into grid cells                 | STING / CLIQUE               |

#### Real Pipeline

```bash
Data → Scaling → K-Means → Cluster Groups → Evaluation
```

#### Clustering Algorithms, Metrics & Model Selection Guide

| Section      | Category        | Algorithm / Metric           | Description                                  | When to Use                                |
| ------------ | --------------- | ---------------------------- | -------------------------------------------- | ------------------------------------------ |
| Algorithms   | Partition-based | K-Means                      | Assigns points to nearest centroid           | Large datasets, simple & fast clustering   |
| Algorithms   | Partition-based | K-Medoids                    | Uses actual data points as cluster centers   | When data has outliers                     |
| Algorithms   | Hierarchical    | Agglomerative                | Bottom-up merging of clusters                | Small datasets, tree structure needed      |
| Algorithms   | Hierarchical    | Divisive                     | Top-down splitting of clusters               | Research/structured analysis               |
| Algorithms   | Density-based   | DBSCAN                       | Finds dense regions, detects noise           | Data with noise/outliers, irregular shapes |
| Algorithms   | Density-based   | OPTICS                       | Improved DBSCAN for varying density          | Complex spatial datasets                   |
| Algorithms   | Model-based     | Gaussian Mixture Model (GMM) | Probabilistic clustering                     | Overlapping clusters, soft grouping        |
| Algorithms   | Model-based     | EM Algorithm                 | Estimates parameters for GMM                 | Probabilistic clustering models            |
| Evaluation   | Metric          | Silhouette Score             | Measures cluster cohesion vs separation      | General clustering quality                 |
| Evaluation   | Metric          | Davies-Bouldin Index         | Measures cluster similarity (lower = better) | Model comparison                           |
| Evaluation   | Metric          | Calinski-Harabasz Index      | Ratio of between vs within cluster variance  | Well-separated clusters                    |
| Evaluation   | Metric          | Inertia                      | Sum of squared distances within clusters     | K-Means optimization                       |
| Evaluation   | Metric          | ARI (Adjusted Rand Index)    | Compares clusters with true labels           | When ground truth exists                   |
| Model Choice | K-Means         | —                            | Spherical, well-separated clusters           | Customer segmentation, market grouping     |
| Model Choice | Hierarchical    | —                            | Tree-based grouping                          | Gene analysis, document clustering         |
| Model Choice | DBSCAN          | —                            | Noise-aware clustering                       | GPS data, fraud detection                  |
| Model Choice | GMM             | —                            | Probabilistic clustering                     | Risk analysis, soft clustering             |

#### Clustering Evaluation Metrics

| Metric                    | Meaning                                                                           | When to Use                           |
| ------------------------- | --------------------------------------------------------------------------------- | ------------------------------------- |
| Silhouette Score          | Measures how well a data point fits within its cluster compared to other clusters | General clustering quality evaluation |
| Davies-Bouldin Index      | Measures similarity between clusters (lower value = better separation)            | Comparing clustering models           |
| Calinski-Harabasz Index   | Ratio of between-cluster variance to within-cluster variance                      | Evaluating cluster separation quality |
| Inertia                   | Sum of squared distances between samples and cluster centroids                    | K-Means optimization (Elbow method)   |
| Adjusted Rand Index (ARI) | Measures similarity between predicted clusters and true labels                    | When ground truth labels exist        |

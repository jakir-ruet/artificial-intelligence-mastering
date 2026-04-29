### Regression

It's a type of supervised learning where the goal is to predict a continuous numeric value based on input features.

> In simple terms: Regression = predicting `how much` or `how many`

**Examples**

| Items       |       Values |
| ----------- | -----------: |
| House price |      250,000 |
| Salary      |       75,000 |
| Temperature |       32.5°C |
| Sales       | 10,500 units |

#### Regression Model Types

| Category          | Model                       | Relationship Type | Description                                            | When to Use                  |
| ----------------- | --------------------------- | ----------------- | ------------------------------------------------------ | ---------------------------- |
| Linear Models     | Linear Regression           | Linear            | Straight-line relationship between features and target | Baseline model, simple data  |
| Linear Models     | Ridge Regression            | Linear            | L2 regularization reduces overfitting                  | Many or correlated features  |
| Linear Models     | Lasso Regression            | Linear            | L1 regularization removes unimportant features         | Feature selection needed     |
| Linear Models     | ElasticNet                  | Linear            | Combination of L1 + L2 regularization                  | Balanced regularization      |
| Tree-Based Models | Decision Tree Regressor     | Non-linear        | Rule-based splits capturing complex patterns           | Interpretable models         |
| Tree-Based Models | Random Forest Regressor     | Non-linear        | Ensemble of trees for better accuracy and stability    | General-purpose strong model |
| Tree-Based Models | Gradient Boosting Regressor | Non-linear        | Sequential error correction                            | High accuracy tasks          |
| Advanced Boosting | XGBoost Regressor           | Non-linear        | Optimized gradient boosting (industry standard)        | Large datasets, competitions |
| Advanced Boosting | LightGBM Regressor          | Non-linear        | Fast and scalable boosting                             | Big data, production systems |
| Distance-Based    | KNN Regressor               | Non-linear        | Predicts based on nearest neighbors                    | Small datasets               |
| Support Vector    | SVR                         | Non-linear        | Margin-based regression with kernel tricks             | Complex but small datasets   |

#### Additional Relationship Concepts

| Type         | Description                |
| ------------ | -------------------------- |
| Linear       | Straight-line relationship |
| Non-linear   | Curved or complex patterns |
| Multivariate | Multiple input features    |

#### Regression Workflow

```bash
Data → Cleaning → Feature Engineering → Scaling → Train → Evaluate → Deploy
```

#### Regression Evaluation Metrics

| Metric   | Full Form                                | Meaning                                                              |
| -------- | ---------------------------------------- | -------------------------------------------------------------------- |
| MAE      | Mean Absolute Error                      | Average absolute difference between actual and predicted values      |
| MSE      | Mean Squared Error                       | Average of squared differences (penalizes large errors more heavily) |
| RMSE     | Root Mean Squared Error                  | Square root of MSE (same unit as target, most commonly used)         |
| R² Score | R-squared (Coefficient of Determination) | Measures how well the model explains variance in the data            |

### Model Development

Model development is the process of creating, training, evaluating, and preparing a machine learning model so it can solve a real-world problem effectively. Think of it like this:

- `Data` → raw material
- `Model development` → turning that raw material into a working intelligent system

#### Typical Workflow

```bash
Data → Validation → Feature Engineering → Training → Evaluation → Optimization → Artifact Storage
```

#### Key Steps

1. Data Loading

	- Load data from CSV, DB, API
	- Ensure schema consistency

2. Data Cleaning & Preprocessing

	- Handle missing values
	- Remove duplicates
	- Encode categorical variables
	- Normalize/scale features

3. Feature Engineering

	- Select important features
	- Create new features
	- Remove irrelevant/noisy data

4. Data Splitting

	- Training set → model learns
	- Validation set → tuning
	- Test set → final evaluation

👉 Common split: 70% Training, 15% Validation, 15% Testing

5. Model Training

	- Select algorithm (e.g., Random Forest, Logistic Regression)
	- Train using training dataset

6. Model Evaluation

	- Evaluate using validation/test data
	- Use proper metrics (not only accuracy)

7. Model Saving (Artifacts)

	- Save trained model
	- Save metadata (parameters, metrics)

### Model Training

Model training is the phase where the model learns patterns from data.

#### Key Concepts

- Train on large, representative dataset
- Avoid overfitting and underfitting
- Use cross-validation for robustness

#### Training Strategy

- Use training dataset to learn
- Validate performance during training
- Test only once (final evaluation)

#### Important Practices

- Fix random seed (reproducibility)
- Avoid data leakage
- Keep training pipeline consistent

### Hyperparameter Tuning

Hyperparameter tuning is the process of finding the best configuration of model settings to maximize performance.

#### Key Idea

- Model Parameters → learned automatically
- Hyperparameters → manually defined before training

#### Common Hyperparameters, like (Random Forest):

- n_estimators
- max_depth
- min_samples_split

#### Tuning Methods

1. Grid Search

	- Tries all combinations
	- Accurate but slow

2. Random Search

	- Tries random combinations
	- Faster and widely used in practice

3. Advanced Methods (Production Level)

	- Bayesian Optimization
	- Optuna
	- Hyperopt

#### Cross-Validation

	- Example: 5-Fold Cross Validation
	- Dataset split into 5 parts
	- Train 5 times, each time with different validation set

#### Evaluation Metrics

In real MLOps, accuracy alone is not enough.

Precision=TP/(FP+TP)
	​
> Measures how many predicted positives are actually correct

Recall=TP/(FN+TP)
	​
> Measures how many actual positives are correctly identified

**Trade-off**
>
> - High Precision → fewer false positives
> - High Recall → fewer false negatives

#### Other Important Metrics

- F1-score (balance of precision & recall)
- ROC-AUC
- Confusion Matrix

#### Example Performance

- Accuracy: 85%
- F1-score: 0.80

> Always evaluate using multiple metrics

### Complete Training Pipeline

1. Data Ingestion
   ↓
2. Data Validation
   ↓
3. Feature Engineering
   ↓
4. Train/Validation Split
   ↓
5. Hyperparameter Tuning
   ↓
6. Train Best Model
   ↓
7. Evaluation (Precision, Recall, F1)
   ↓
8. Save Model (Artifact)

### Final Summary

- `Model Development` = building and improving ML models
- `Model Training` = learning from data
- `Hyperparameter Tuning` = optimizing model performance
- `Evaluation Metrics` = measuring real performance

### ML Flow

MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It helps data scientists and ML engineers keep track of experiments, package code, and deploy models in a consistent way.

![ML Flow](/img/ml-flow.png)

#### Core Components of MLflow

**1. MLflow Tracking**

- Logs experiments: parameters, metrics, artifacts (models, plots, etc.)
- Lets you compare runs visually
- Works with local files or remote servers

```bash
import mlflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

**2. MLflow Projects**

- Standardizes how ML code is packaged and run
- Uses a simple format (MLproject file)
- Makes experiments reproducible

**3. MLflow Models**

- Packages models in a standard format
- Supports multiple flavors (scikit-learn, TensorFlow, PyTorch, etc.)
- Enables easy deployment across platforms

**4. MLflow Model Registry**

- Central hub for managing models
- Tracks versions, stages (Staging, Production)
- Supports collaboration and governance

#### Why Use MLflow?

- Keeps experiments organized
- Makes models reproducible
- Simplifies deployment
- Framework-agnostic (works with most ML libraries)
- Scales from local development to production

#### Simple Workflow

- Train a model
- Log results with MLflow Tracking
- Package it using MLflow Models
- Register it in Model Registry
- Deploy or serve the model

### Classification

Classification is a supervised learning technique where the model predicts a category (class label). **Output is discrete, not continuous.**

**Examples**

| Item         | Description        |
| ------------ | ------------------ |
| Email        | Spam/Not Spam      |
| Bank loan    | Approved/Rejected  |
| Medical test | Disease/No Disease |
| Image        | Cat/Dog/Horse      |

#### Classification Types

| Type                       | Description                                             | Example                                                                       |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Binary Classification      | Only **2 classes**                                      | Yes/No  <br> True/False  <br> Spam/Not Spam                                   |
| Multi-class Classification | **More than 2 classes**, but only **one correct label** | Dog/Cat/Horse  <br> Digit recognition (0–9)  <br> Red/Green/Blue              |
| Multi-label Classification | **Multiple labels at the same time**                    | Image → “Dog + Outdoor + Daytime”  <br> Movie → “Action + Comedy + Adventure” |

> **Quick Memory Trick**
>
> - Binary → 2 choices
> - Multi-class → 1 of many
> - Multi-label → many at once

#### Classification Workflow

1. Collect Data
2. Clean Data
3. Preprocess (encode, scale)
4. Split dataset (train/test)
5. Choose model
6. Train model (fit)
7. Predict
8. Evaluate
9. Improve (tuning)

#### Classification Algorithms in Scikit-learn

| Category       | Model                        | Description                                                                     | Example Use Case                             |
| -------------- | ---------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------- |
| Linear Models  | Logistic Regression          | Most important baseline model; uses linear decision boundary for classification | Spam detection, medical diagnosis (Yes/No)   |
| Tree Models    | Decision Tree                | Splits data using rules in a tree structure                                     | Loan approval, simple decision systems       |
| Tree Models    | Random Forest                | Ensemble of many decision trees; more accurate and stable                       | Fraud detection, customer churn prediction   |
| Distance-based | K-Nearest Neighbors (KNN)    | Classifies based on nearest data points in feature space                        | Recommendation systems, image classification |
| Margin-based   | Support Vector Machine (SVM) | Finds the best boundary (hyperplane) that separates classes                     | Face detection, text classification          |
| Probabilistic  | Naive Bayes                  | Uses probability based on feature independence assumption                       | Spam filtering, sentiment analysis           |

#### When to Use Which Model

| Model               | Best For                      |
| ------------------- | ----------------------------- |
| Logistic Regression | Simple baseline               |
| Naive Bayes         | Text/spam detection           |
| Decision Tree       | Interpretable logic           |
| Random Forest       | Strong general model          |
| SVM                 | High accuracy on complex data |
| KNN                 | Small datasets                |

#### Real Pipeline

```bash
Raw Data → Cleaning → Feature Engineering → Model → Evaluation → Deployment
```

#### Classification Evaluation Metrics

| Metric    | Full Form                                            | Meaning                                                     | When to Use                         |
| --------- | ---------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------- |
| Accuracy  | Accuracy Score                                       | Percentage of correct predictions                           | When dataset is balanced            |
| Precision | Precision Score                                      | Correct positive predictions out of all predicted positives | When false positives are costly     |
| Recall    | Recall (Sensitivity)                                 | Correct positives out of all actual positives               | When missing positives is costly    |
| F1 Score  | F1 Score                                             | Harmonic mean of Precision and Recall                       | For imbalanced datasets             |
| ROC-AUC   | Receiver Operating Characteristic - Area Under Curve | Measures model’s ability to distinguish between classes     | Binary classification problems      |
| Log Loss  | Logarithmic Loss                                     | Penalizes confident wrong predictions                       | Probabilistic classification models |

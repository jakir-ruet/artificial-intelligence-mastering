## More About Me – [Take a Look!](http://www.mjakaria.me)

### Overview

Machine learning and AI are rapidly growing fields. This course will teach you the fundamentals of machine learning, including the workflow and real-world challenges. This will give you the foundation for ML discussions and continuing the learning.

In Simple terms

| Traditional programming                                           | Machine Learning                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| You give the computer `rules` + `data` → it produces an `answer`. | You give the computer `data` + `answers` → it learns the `rules`. |

- `Machine Learning in Action`
  - Image recognition
  - Fraud detection
  - Speech recognition
  - Spam detection

- `A Few Core Tasks`
  - `Classification` > Assigns data to predefined `categories` or `labels`. i.e; Determining whether an email is `spam` or `not spam`.
  - `Regression` > Predicts a continuous `numerical value`. i.e; Estimating `house prices` based on `size`, `location`, and `features`.
  - `Clustering` > Groups similar `data points` together without predefined labels. i.e; Segmenting customers into `behavioral groups` for marketing.
  - `Recommendation` > Suggest items or content based on user `behavior` or `preferences`. i.e; Recommending `movies` on a streaming platform based on what you’ve watched.

- `Deploying Machine Learning`
  - `Accurate` > The model should make predictions that are correct and reliable in real-world conditions.
  - `Explainable` > Its decisions should be understandable to humans—important for trust, debugging, and compliance.
  - `Fast` > Predictions need to be made quickly enough to meet user or system requirements.
  - `Fair` > The model must avoid harmful bias and perform equitably across different groups.
  - `Scalable` > It should handle growing amounts of data and users without performance issues.
  - `Maintainable` > The system should be easy to update, monitor, and improve over time as data and requirements change.

- `Accuracy`
  - Of all predictions made, how many were correct?
  - If it’s right `999/1000` times > `99.9%` accuracy
  - Sometimes, accuracy doesn’t tell the whole story

`Accuracy` = `(𝑇𝑃 + 𝑇𝑁)`/`(𝑇𝑃 + 𝑇𝑁 + 𝐹𝑃 + 𝐹𝑁)`

Where,
- `TP`: True Positive (correctly predicted `yes`)
- `TN`: True Negative (correctly predicted `no`)
- `FP`: False Positive (incorrectly predicted `yes`)
- `FN`: False Negative (missed a `yes`)

> Works well when classes are balanced. Misleading for imbalanced data (e.g., 99% non-fraud, 1% fraud).

- `Precision` > How often is a positive prediction correct?

`Precision` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑃)`

> High precision = few false alarms.

- `Recall` > How often is a real positive correctly predicted?

`Recall` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑁)`

> High recall = you catch most of the positive cases.

- `F1 Score` > Harmonic mean of `precision` and `recall`.

`𝐹1` = 2 × (`(Precision × Recall)`/`(Precision + Recall)`)

> Useful when you want a balance between precision and recall.

- `Predicting Machine Maintenance`

| Strategy               | Precision | Recall   | Note                                 |
| ---------------------- | --------- | -------- | ------------------------------------ |
| Predict all machines   | Low       | High     | Catch all potential issues           |
| Predict only when sure | High      | Low      | Avoid unnecessary maintenance        |
| F1 score               | Balanced  | Balanced | Trade-off between precision & recall |

- `Latency` > How quickly the model can make predictions. Especially important for real-time systems- E.g., a system that detects danger

- `Key considerations for deploying and evaluating machine learning`
  - `Use Cases:` > Domains where ML is applied (e.g., self-driving cars, intrusion detection, medical diagnosis).
  - `Scalability:` > Ability to handle more data, users, or requests without performance loss.
  - `Fairness:` > Ensuring predictions are equitable across all groups.
  - `Explainability:` > Understanding and communicating why a prediction was made.
  - `Reliability:` > Consistency and robustness over time; reproducible results.

### Machine Learning Lifecycle

1. `Problem Formulation`
   - Define the problem, objectives, and success metrics.
2. `Data Collection`
   - Gather relevant, high-quality data from multiple sources.
3. `Data Preparation`
   - Clean data, engineer features, split into train/validation/test sets.
4. `Model Training & Evaluation`
   - Select algorithm, train model, evaluate performance, tune hyperparameters.
5. `Model Deployment`
   - Integrate model into production for predictions (APIs, dashboards, pipelines).
6. `Monitoring & Maintenance`
   - Track performance, detect drift, retrain as needed.
7. `Governance & Reproducibility`
   - Document steps, ensure reproducibility, enforce compliance and security.

![Machine Learning Lifecycle](/img/ml-lifecyle.png)

### Let's explore individual component - Machine Learning Lifecycle

#### 1. `Problem Formulation`

- What type of problem is this?
  Is it a classification, regression, clustering, or reinforcement learning problem?
- What kind of data do you have?
  Is it structured (tabular), unstructured (images, text, audio), or time series data?
- What are the performance metrics or objectives?
  Are you optimizing for accuracy, precision, recall, F1-score, AUC, RMSE, etc.?

#### 2. `Data Collection`

- Identifying relevant data sources and pulling it into your system-
  - APIs
  - Internal databases
  - Web scraping
  - Third-party partners
- In enterprise settings, often involves
  - Working with stakeholders
  - Understanding data availability and permissions

#### 3. `Data Preparation`

- Cleaning
- Normalization and encoding
- Feature extraction and transformation
- Handling sensitive data
- Splitting the dataset

#### 4. `Model Training & Evaluation`

- `Depends on`
  - The type of problem
  - The characteristics of your data
  - Trade-offs you’re willing to accept:
  - Interpretability,
  - Performance, and
  - Resource usage

- `Model Type`
  - Linear regression
  - Decision trees
  - Random forests
  - Gradient boosting machines
  - Support vector machines
  - Neural networks

- `Summary Comparison`

| **Model**             | **Type**                  | **Pros**                             | **Cons**                             | **When to Use**                     |
| --------------------- | ------------------------- | ------------------------------------ | ------------------------------------ | ----------------------------------- |
| **Linear Regression** | Regression                | Simple, interpretable                | Assumes linearity                    | Simple linear problems              |
| **Gradient Boosting** | Classification/Regression | High performance                     | Slow, hard to interpret              | Complex tasks needing high accuracy |
| **Decision Trees**    | Classification/Regression | Interpretable, handles non-linearity | Overfitting, unstable                | Need interpretability               |
| **SVM**               | Classification/Regression | Works in high dimensions             | Slow, hard to interpret              | High-dimensional data               |
| **Random Forests**    | Classification/Regression | Reduces overfitting                  | Slow predictions, less interpretable | High accuracy needed                |
| **Neural Networks**   | Classification/Regression | Models complex patterns              | Requires large data                  | Image, text, speech tasks           |

- `Evaluation Metrics`

| **Metric**    | **What it Measures**                 | **Question it Answers**                                     | **Formula**                                                |
| ------------- | ------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------- |
| **Accuracy**  | Overall correctness of the model     | Of all predictions, how many were correct?                  | `Accuracy` = `(𝑇𝑃 + 𝑇𝑁)`/`(𝑇𝑃 + 𝑇𝑁 + 𝐹𝑃 + 𝐹𝑁)`             |
| **Precision** | Correctness of positive predictions  | When the model predicts positive, how often is it correct?  | `Precision` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑃)`                             |
| **Recall**    | Ability to detect actual positives   | How many actual positives did the model correctly identify? | `Recall` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑁)`                                |
| **F1 Score**  | Balance between Precision and Recall | What is the harmonic mean of precision and recall?          | `𝐹1` = 2 × (`(Precision × Recall)`/`(Precision + Recall)`) |

> Imbalanced Classes: Why Accuracy Can Be Misleading

When dealing with datasets where one class is much rarer than the other, accuracy becomes a poor performance metric, Example

- Only 1 out of 100 candidates is successful.
- A model that predicts everyone as unsuccessful still gets 99% accuracy
- But it completely misses the one successful case
- Precision and recall are more helpful in these situations

- `Better Metrics in Imbalanced Settings`
  - Precision: Of the cases predicted positive, how many are correct?
  - Recall: Of all actual positive cases, how many did the model detect?
  - F1 score: Harmonic mean of precision and recall.
  - ROC-AUC / PR-AUC: Useful alternative metrics for imbalance.

> Hyperparameters

- Settings that control how the algorithm behaves, They are not learned from the data, Example;
- Max depth in a decision tree
  - Controls how many levels the tree can split
  - Too shallow? May miss patterns
  - Too deep? May overfit and memorize the training data

> Cross-Validation

Cross-validation checks whether a model generalizes well to new, unseen data.

- Split the data into k subsets (folds).
- Train on k-1 folds, test on the remaining fold.
- Repeat until each fold has been used for testing.
- Average the results for a more reliable performance estimate.

- `Goal`
  - Avoiding overfitting
  - Ensuring the model performs well not only on the training data but also on unseen data.

> Overfitting

When a model learns the training data too well and performs poorly on new data.

> Underfitting

When a model is too simple to capture important patterns.

#### 5. `Model Deployment`

- The process of making a model accessible to whoever or whatever needs it
- Typically done through a REST API
- The model itself is usually packaged and hosted on a server or in the cloud
- Common tools
  - Docker,
  - Kubernetes
  - AWS SageMaker
  - Azure ML
  - Google Vertex AI
- Real-Time Inference in Model Deployment
![Real-Time Inference in Model Deployment](/img/real-inference-model-deploy.png)

- Batch Inference in Model Deployment
  - Used when predictions don’t need to happen instantly
  - Predictions are run periodically on batches of inputs
  - The model is deployed inside a scheduled process
  - Runs predictions on large amounts of data at once
  - Results are typically saved for later use

#### 6. `Monitoring & Maintenance`

 Continuously looking at data to see if your model is still working well after it’s been deployed

- Infrastructure for logging and observability
  - Track which inputs the model receives
  - Log the outputs returned by the model
  - Measure prediction latency
  - Monitor for any error

- Using Logs and Data to Evaluate Model Health
  - Is the model still performing well?
  - Is it making predictions within the expected latency?
  - Has the input data changed significantly?
  - Are certain outputs becoming skewed or biased over time?

- Metrics for Monitoring Deployed Models
  - Prediction distribution > Are the scores shifting?
  - Data drift > Are the incoming inputs different from the training data?
  - Concept drift > Has the underlying relationship between input and output changed?
  - Latency and throughput > Are we meeting performance targets?

#### 7. `Governance & Reproducibility`

- In traditional software, version control is a given
  - You always know what code is running and
  - how to roll back if needed
- ML needs same discipline, but more complex
- We must track not just code, but also
  - The data
  - The model version
  - The training process
  - The hyperparameters
  - The runtime environment

## Wth Regards, `Jakir`

[![LinkedIn][linkedin-shield-jakir]][linkedin-url-jakir]
[![Facebook-Page][facebook-shield-jakir]][facebook-url-jakir]
[![Youtube][youtube-shield-jakir]][youtube-url-jakir]

### Wishing you a wonderful day! Keep in touch.

<!-- Personal profile -->

[linkedin-shield-jakir]: https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url-jakir]: https://www.linkedin.com/in/jakir-ruet/
[facebook-shield-jakir]: https://img.shields.io/badge/Facebook-%231877F2.svg?style=for-the-badge&logo=Facebook&logoColor=white
[facebook-url-jakir]: https://www.facebook.com/jakir.ruet/
[youtube-shield-jakir]: https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white
[youtube-url-jakir]: https://www.youtube.com/@mjakaria-ruet/featured

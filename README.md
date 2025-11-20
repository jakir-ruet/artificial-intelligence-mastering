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

#### TalentFlow Example - Machine Learning Lifecycle

1. `Problem Formulation`
   - Define what we are predicting (e.g., a “good hire”).
   - Example criteria:
     - Stays longer than 1 year
     - Performs well in evaluations
   - Goal: Determine the prediction target.
2. `Data: Collection & Preparation`
   - **Foundation of any ML system.**
   - Not just getting data, but **right and usable data**.
   `Data Collection`
   - Identify sources: APIs, internal databases, web scraping, third-party partners.
   - Enterprise considerations: stakeholder collaboration, data availability, permissions.
   - **TalentFlow example:**
   - Historical resumes, interview notes, job descriptions, retention metrics, performance reviews.
   `Supervised Learning`
   - Requires labeled data (input + expected output).
   - **TalentFlow example:** Resume + performance score.
   `Data Preparation`
   - Cleaning: remove duplicates, handle typos, fill missing values, anonymize personal info.
   - Transformation: NLP on resumes, numeric scoring for interview notes, encoding retention labels.
   - Splitting dataset: Training / Validation / Test sets.
   - **Important:** Each step can introduce subtle issues; preparation is ongoing, not one-time.
3. `Data Pipelines`
- Ensure **consistency** and **scalability**.
- Tools: Apache Airflow, dbt, Python scripts.
4. `Effects of Poor Data Preparation`
- Model may fail silently in production.
- Garbage in → Garbage out.
- Poor input leads to poor predictions and bad decisions.

### Model Development & Training

Model development and training is a critical part of building machine learning systems. Choosing the right model is one of the most important steps in the process. Here's a breakdown of the general approach to model selection:

#### Understanding the Problem

- What type of problem is this?
  Is it a classification, regression, clustering, or reinforcement learning problem?
- What kind of data do you have?
  Is it structured (tabular), unstructured (images, text, audio), or time series data?
- What are the performance metrics or objectives?
  Are you optimizing for accuracy, precision, recall, F1-score, AUC, RMSE, etc.?

#### Choosing a Model Type

- Depends on:-
  - The type of problem
  - The characteristics of your data
  - Trade-offs you’re willing to accept:
  - Interpretability,
  - Performance, and
  - Resource usage

##### Machine Learning Model

- Linear regression
- Decision trees
- Random forests
- Gradient boosting machines
- Support vector machines
- Neural networks

- Summary Comparison

| **Model**             | **Type**                  | **Pros**                             | **Cons**                             | **When to Use**                     |
| --------------------- | ------------------------- | ------------------------------------ | ------------------------------------ | ----------------------------------- |
| **Linear Regression** | Regression                | Simple, interpretable                | Assumes linearity                    | Simple linear problems              |
| **Gradient Boosting** | Classification/Regression | High performance                     | Slow, hard to interpret              | Complex tasks needing high accuracy |
| **Decision Trees**    | Classification/Regression | Interpretable, handles non-linearity | Overfitting, unstable                | Need interpretability               |
| **SVM**               | Classification/Regression | Works in high dimensions             | Slow, hard to interpret              | High-dimensional data               |
| **Random Forests**    | Classification/Regression | Reduces overfitting                  | Slow predictions, less interpretable | High accuracy needed                |
| **Neural Networks**   | Classification/Regression | Models complex patterns              | Requires large data                  | Image, text, speech tasks           |

##### Evaluation Metrics

| **Metric**    | **What it Measures**                 | **Question it Answers**                                     | **Formula**                                                |
| ------------- | ------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------- |
| **Accuracy**  | Overall correctness of the model     | Of all predictions, how many were correct?                  | `Accuracy` = `(𝑇𝑃 + 𝑇𝑁)`/`(𝑇𝑃 + 𝑇𝑁 + 𝐹𝑃 + 𝐹𝑁)`             |
| **Precision** | Correctness of positive predictions  | When the model predicts positive, how often is it correct?  | `Precision` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑃)`                             |
| **Recall**    | Ability to detect actual positives   | How many actual positives did the model correctly identify? | `Recall` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑁)`                                |
| **F1 Score**  | Balance between Precision and Recall | What is the harmonic mean of precision and recall?          | `𝐹1` = 2 × (`(Precision × Recall)`/`(Precision + Recall)`) |

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

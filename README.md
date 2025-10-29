## More About Me – [Take a Look!](http://www.mjakaria.me)

### Roadmap of Artificial Intelligence (AI)

**Comprehensive roadmap to learn Artificial Intelligence (AI)** — from beginner to pro, structured in clear phases with recommended topics, tools, and resources.

![AI Model](/img/ai-model.png)

#### OVERVIEW

AI is a broad field. The core components you'll need to master include:

- 🧮 Mathematics
- 💻 Programming
- 🤖 Machine Learning (ML)
- 🧠 Neural Networks (NN)
- 🔍 Deep Learning (DL)
- 🧰 Tools & Frameworks
- 💼 Projects
- 🚀 Advanced Topics (NLP, Computer Vision, Reinforcement Learning)

#### Mathematics for ML/DL

Build core intuition to understand how algorithms work.

| **Topic**                    | **Key Concepts**                            | **Resources**                                                                                                                          |
| ---------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Linear Algebra**           | Vectors, matrices, dot product, eigenvalues | [3Blue1Brown’s Linear Algebra](https://www.youtube.com/c/3blue1brown), [Khan Academy](https://www.khanacademy.org/math/linear-algebra) |
| **Calculus**                 | Derivatives, gradients, chain rule          | [Khan Academy](https://www.khanacademy.org/math/calculus-1), Visual Calculus                                                           |
| **Probability & Statistics** | Distributions, Bayes’ Theorem, expectation  | [StatQuest](https://www.youtube.com/user/joshstarmer), Khan Academy                                                                    |
| **Optimization**             | Gradient descent, cost functions            | DeepLearning.AI specialization, ML blogs                                                                                               |

#### Programming

Be comfortable writing and debugging Python code for data and ML.

| **Skill**               | **Tools/Libraries**                | **Resources**                                                                                                                                         |
| ----------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python**              | Basic syntax, data structures, OOP | [Codecademy](https://www.codecademy.com/learn/learn-python-3), [Real Python](https://realpython.com/), [W3Schools](https://www.w3schools.com/python/) |
| **NumPy, Pandas**       | Data manipulation, arrays          | [Kaggle Python Course](https://www.kaggle.com/learn/python), [DataCamp](https://www.datacamp.com/)                                                    |
| **Matplotlib, Seaborn** | Data visualization                 | YouTube, DataCamp                                                                                                                                     |
| **Jupyter Notebooks**   | Interactive coding environment     | Use in Google Colab or VS Code                                                                                                                        |
| **Git & GitHub**        | Version control, collaboration     | [GitHub Learning Lab](https://lab.github.com/), [freeCodeCamp](https://www.freecodecamp.org/)                                                         |

#### Machine Learning (ML)

Machine Learning is a branch of artificial intelligence (AI) that enables computers to learn patterns and make decisions or predictions from data without being explicitly programmed. Instead of following fixed rules, machine learning algorithms automatically improve their performance over time as they are exposed to more data and experience. To develop a solid understanding of `machine learning`, we should have a good grasp of the following key topics.

- Machine Learning Topics

![Machine Learning Workflow](/img/ml-topics.jpg)

#### Neural Networks (NN)

To develop a solid understanding of `Neural Networks`, we should have a good grasp of the following key topics.

| **Concept**              | **What to Learn**                    |
| ------------------------ | ------------------------------------ |
| **Neurons & Layers**     | Input, hidden, and output layers     |
| **Forward Propagation**  | How data flows through the network   |
| **Backward propagation** | Gradient descent, error minimization |
| **Activation Functions** | ReLU, Sigmoid, Tanh                  |
| **Loss Functions**       | MSE, Cross-Entropy                   |

#### Deep Learning (DL)

Deep Learning is a specialized subfield of machine learning that focuses on algorithms inspired by the structure and function of the human brain, known as artificial neural networks. It involves training these multi-layered networks (called deep neural networks) to automatically learn hierarchical patterns and representations from large amounts of data. To develop a solid understanding of `deep learning`, we should have a good grasp of the following key topics.

- Deep Learning Topics

![Deep Learning Workflow](/img/dl-topics.jpg)

#### Complete Guideline: Building an ML/AI Application (Start → Deployment)

##### 1. Define the Problem Clearly > Understand what problem you’re solving.

**Steps:**
- Identify the business or real-world problem.
- Define input and output (what data you have and what you want to predict).
- Choose measurable success metrics (accuracy, RMSE, precision, etc.).
- Check if machine learning is the right solution.

**Example:**
Predict whether a customer will churn based on usage data.

##### 2. Collect and Gather Data > Obtain high-quality, relevant data.

**Sources:**
- Company databases (SQL, data warehouses)
- APIs or web scraping
- Public datasets (Kaggle, UCI, Google Dataset Search)
- IoT devices or sensors

**Tips:**
- Ensure data privacy (GDPR, HIPAA)
- Gather enough data to train the model effectively

##### 3. Data Cleaning and Preprocessing > Prepare raw data for analysis and modeling.

**Steps:**
- Handle missing values, outliers, and duplicates
- Encode categorical variables (Label/One-Hot Encoding)
- Normalize or standardize numerical features
- Split dataset:
  - **Train (70%)**
  - **Validation (15%)**
  - **Test (15%)**

**Tools:** `pandas`, `NumPy`, `scikit-learn`

##### 4. Exploratory Data Analysis (EDA) > Understand data structure and key relationships.

**Steps:**
- Visualize distributions, correlations, and trends
- Identify data patterns and anomalies
- Understand which features influence the target variable

**Tools:** `matplotlib`, `seaborn`, `plotly`

**Outcome:** Insights that guide feature selection and model choice

##### 5. Feature Engineering > Improve model performance with better features.

**Steps:**
- Create new features from existing ones
- Select important features (feature importance, correlation)
- Apply dimensionality reduction (PCA, feature selection)

**Outcome:** A clean, optimized feature set

##### 6. Model Selection and Training > Choose and train the most suitable ML algorithm.

**Steps:**
- Identify the problem type:
  - **Classification:** Logistic Regression, Random Forest, XGBoost
  - **Regression:** Linear Regression, Decision Tree, Gradient Boosting
  - **Clustering:** K-Means, DBSCAN
- Train multiple models and compare performance
- Tune hyperparameters (Grid Search, Random Search, Bayesian Optimization)

**Tools:** `scikit-learn`, `TensorFlow`, `PyTorch`, `XGBoost`, `LightGBM`

##### 7. Model Evaluation > Test model performance on unseen data.

**Common Metrics:**
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression:** MAE, MSE, RMSE, R²
- **Clustering:** Silhouette Score, Davies–Bouldin Index

**Steps:**
- Evaluate on the test set
- Check for overfitting/underfitting
- Analyze errors and misclassifications

##### 8. Model Optimization and Validation > Refine model performance and ensure robustness.

**Steps:**
- Fine-tune hyperparameters
- Use ensemble techniques (bagging, boosting, stacking)
- Cross-validate results for consistency
- Retrain with optimized settings

##### 9. Model Packaging > Prepare your model for deployment.

**Steps:**
- Save trained model (`.pkl`, `.joblib`, `.pt`, `.h5`)
- Build a prediction pipeline (input → preprocess → model → output)
- Create an API for prediction using Flask or FastAPI
- Test locally

**Tools:** `Flask`, `FastAPI`, `Docker`, `Pickle`, `MLflow`

##### 10. Deployment > Make your model available for real users or systems.

**Deployment Options:**
- **Web App:** Flask/FastAPI + Streamlit/Gradio for UI
- **Cloud Platforms:** AWS Sagemaker, Google Vertex AI, Azure ML
- **Containers:** Docker or Kubernetes for scalability
- **Edge Devices:** TensorFlow Lite, ONNX for mobile or IoT

**Best Practices:**
- Use version control (`Git`)
- Automate with CI/CD pipelines
- Secure endpoints (authentication, HTTPS)

##### 11. Monitoring and Maintenance > Ensure model performance remains stable post-deployment.

**Steps:**
- Monitor prediction accuracy and latency
- Detect data drift and model degradation
- Log inputs/outputs for feedback
- Retrain periodically with new data

**Tools:** `MLflow`, `Prometheus`, `Grafana`, `Evidently AI`

##### 12. Continuous Improvement > Keep enhancing the ML system over time.

**Steps:**
- Gather feedback from users
- Update data and retrain regularly
- Improve features or try advanced models
- Document every step for reproducibility

##### Summary Workflow

| Step | Description            | Tools                     |
| ---- | ---------------------- | ------------------------- |
| 1    | Define Problem         | —                         |
| 2    | Collect Data           | SQL, APIs, Kaggle         |
| 3    | Clean & Prepare Data   | pandas, NumPy             |
| 4    | EDA                    | seaborn, matplotlib       |
| 5    | Feature Engineering    | scikit-learn              |
| 6    | Model Training         | scikit-learn, PyTorch     |
| 7    | Evaluation             | metrics, confusion matrix |
| 8    | Optimization           | Optuna, GridSearchCV      |
| 9    | Packaging              | Flask, FastAPI            |
| 10   | Deployment             | AWS, Docker, Kubernetes   |
| 11   | Monitoring             | MLflow, Evidently         |
| 12   | Continuous Improvement | CI/CD, retraining         |

##### Simple Flow:

**Problem → Data → Preparation → Modeling → Evaluation → Deployment → Monitoring → Improvement**

#### Tools & Frameworks

Build, train, and deploy ML/DL models efficiently.

| **Tool**                       | **Use**                                       |
| ------------------------------ | --------------------------------------------- |
| **Scikit-learn**               | Classical ML models                           |
| **TensorFlow / Keras**         | Deep Learning models (user-friendly)          |
| **PyTorch**                    | Deep Learning models (more control, flexible) |
| **Google Colab**               | Free GPU-powered notebooks                    |
| **Hugging Face Transformers**  | Pre-trained NLP models                        |
| **Flask / Streamlit / Docker** | Model deployment and versioning               |
| **OpenCV, YOLO**               | Computer vision applications                  |

#### Category-wise Toolset

| **Category**             | **Tools**                        |
| ------------------------ | -------------------------------- |
| **ML**                   | Scikit-learn, Pandas, Matplotlib |
| **DL**                   | TensorFlow, Keras, PyTorch       |
| **NLP**                  | spaCy, Hugging Face Transformers |
| **CV (Computer Vision)** | OpenCV, YOLO, PyTorch            |
| **Deployment**           | Flask, Streamlit, Docker         |

#### Projects

Apply your skills to real-world data and build a strong portfolio.

| **Type**                   | **Examples**                                        |
| -------------------------- | --------------------------------------------------- |
| **ML**                     | Titanic survival prediction, house price prediction |
| **DL - CV**                | Dog vs. cat classifier, facial recognition          |
| **DL - NLP**               | Sentiment analysis, chatbot, text summarizer        |
| **Reinforcement Learning** | Game-playing AI (CartPole, Atari)                   |

#### Advanced Topics

Dive deeper into specialized AI domains.

| **Topic**                             | **Focus Area**                           |
| ------------------------------------- | ---------------------------------------- |
| **Natural Language Processing (NLP)** | Transformers, BERT, GPT, text generation |
| **Computer Vision (CV)**              | Object detection, segmentation, OpenCV   |
| **Reinforcement Learning (RL)**       | Q-learning, DQN, policy gradients        |
| **MLOps**                             | Model deployment, versioning, monitoring |
| **Self-supervised Learning**          | Modern unsupervised deep learning        |
| **Graph Neural Networks (GNNs)**      | Node classification, link prediction     |

#### Recommended Platforms & Resources

- [Kaggle Python Course](https://www.kaggle.com/learn/python)
- [Google Colab](https://colab.research.google.com/)
- [Harvard CS50’s AI Course](https://cs50.harvard.edu/ai/)
- [Visual Calculus](https://mathinsight.org/calculus)
- [3Blue1Brown’s Linear Algebra Series](https://www.youtube.com/c/3blue1brown)

#### Final Tip

> Build projects, share them on GitHub, and join AI communities.
> Learning AI is a marathon, not a sprint — stay consistent and curious!

## With Regards, `Jakir`

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

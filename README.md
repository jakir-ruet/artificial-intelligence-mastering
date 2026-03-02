## More About Me – [Take a Look!](http://www.mjakaria.me)

### Roadmap of Artificial Intelligence (AI)

`Comprehensive roadmap to learn Artificial Intelligence (AI)` — from beginner to pro, structured in clear phases with recommended topics, tools, and resources.

![AI Model](/img/ai-model.png)

#### OVERVIEW

AI is a broad field. The core components you'll need to master include:

- 🧮 Mathematics for Machine Learning
- 💻 Programming in Python
- 🤖 Machine Learning (ML)
- 🧠 Neural Networks (NN)
- 🔍 Deep Learning (DL)
- 🧰 Tools & Frameworks
- 💼 Projects
- 🚀 Advanced Topics (NLP, Computer Vision, Reinforcement Learning)

#### Mathematics for Machine Learning

1. `Linear Algebra:` Vectors, matrices, dot product, eigenvalues.
2. `Probability & Statistics:` Distributions, Bayes’ Theorem, expectation.
3. `Calculus Basics:` Derivatives, gradients, chain rule.
4. `Optimization:` Gradient descent, cost functions

#### Programming in Python-[Visit](https://github.com/jakir-ruet/mastering-with-python)

Be comfortable writing and debugging Python code for data and ML.

1. Python Syntax, Python Comments, Python Variables, Python Data Types, Python Numbers, Python Casting, Python Strings, Python Booleans, Python Operators.
2. Python Lists, Python Tuple, Python Sets, Python Dictionaries Python If...Else, Python Match.
3. Python While Loops, Python for Loops, Python Functions, Python Lambda.
4. Python Classes/Objects, Python Inheritance, Python Iterators Python Polymorphism.
5. Python Polymorphism, Python Modules Python Dates, Python Math.
6. Python JSON, Python RegEx, Python PIP, Python Try...Except.
7. Python String Formatting, Python User Input, Python VirtualEnv.
8. Python File Handling, Python Read Files, Python Write/Create Files Python Delete Files.

##### Python Library (Programming for ML and DL)

1. `Matplotlib:` (creating static, animated, and interactive plots).
2. `Seaborn:` (high-level visualization library)
3. `SciPy:` (scientific computing library built on NumPy)
4. `Pandas:` (Data Frames Handles labeled data)
5. `NumPy:` Handles numerical data (arrays, matrices).

#### Machine Learning (ML)

Machine Learning is a branch of artificial intelligence (AI) that enables computers to learn patterns and make decisions or predictions from data without being explicitly programmed. Instead of following fixed rules, machine learning algorithms automatically improve their performance over time as they are exposed to more data and experience.

In Simple terms

| Traditional programming                                           | Machine Learning                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| You give the computer `rules` + `data` → it produces an `answer`. | You give the computer `data` + `answers` → it learns the `rules`. |

##### How Machine Learning Works (at a high level)

1. Collect Data > (e.g., customer info, images, text)
2. Prepare Data > Clean, normalize, and split into training/testing sets.
3. Choose a Model > (e.g., linear regression, decision tree, neural network)
4. Train the Model > Feed it data so it learns patterns.
5. Test the Model > Evaluate accuracy on new (unseen) data.
6. Use the Model > Make predictions or decisions automatically.

##### Main Types of Machine Learning

| **Type**                   | **What it does**                              | **Example**                                  |
| -------------------------- | --------------------------------------------- | -------------------------------------------- |
| **Supervised Learning**    | Learns from labeled data (has right answers). | Predict house prices, classify emails.       |
| **Unsupervised Learning**  | Finds patterns in unlabeled data.             | Customer segmentation, topic grouping.       |
| **Reinforcement Learning** | Learns by trial and error using rewards.      | Training robots, self-driving cars, game AI. |

| **Learning Type**          | **Student Analogy**                                                                           | **Why**                                                                    |
| -------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Supervised Learning**    | A student solving math problems **with an answer sheet**                                      | The student can compare answers and learn from mistakes (data has labels). |
| **Unsupervised Learning**  | A student solving math problems **with no answer sheet**                                      | The student must find patterns or relationships alone (no labels).         |
| **Reinforcement Learning** | A student learning through **rewards and penalties** (points, grades, feedback after actions) | The student improves behavior based on success or failure outcomes.        |

> Simple memory trick

- `Supervised` → Teacher shows answers
- `Unsupervised` → Student discovers patterns
- `Reinforcement` → Learn by reward & punishment

##### Machine Learning frameworks and model persistence tools

1. `XGBoost (Extreme Gradient Boosting):` A high-performance gradient boosting library for supervised learning (especially tabular data).
2. `LightGBM (Light Gradient Boosting Machine):` A gradient boosting framework by Microsoft - optimized for speed and low memory.
3. `Joblib:` A tool for serializing (saving/loading) Python objects efficiently, especially large NumPy arrays or scikit-learn models.
4. `Pickle:` A standard Python module for serializing and deserializing Python objects.
5. `Optuna:` An automatic hyperparameter optimization library.
6. `TensorFlow/Keras:` Google’s deep learning framework. Keras is now the official high-level API inside TensorFlow (tf.keras).
7. `PyTorch/Torchvision:` Meta’s (Facebook’s) deep learning framework.
8. `FastAI:` A high-level deep learning library built on top of PyTorch.

##### How they fit together

| `Library`               | `Type`         | `Built On`      | `Main Use`                           |
| ----------------------- | -------------- | --------------- | ------------------------------------ |
| `Optuna`                | Optimization   | Independent     | Hyperparameter tuning                |
| `TensorFlow / Keras`    | Deep learning  | TensorFlow core | Neural networks, production models   |
| `PyTorch / Torchvision` | Deep learning  | PyTorch core    | Research, flexible modeling          |
| `FastAI`                | High-level API | PyTorch         | Quick prototyping, transfer learning |

##### Data Preprocessing?

Data preprocessing means cleaning, transforming, and organizing raw data so that it can be efficiently used by a machine learning model.

- Raw data usually has:
  - Missing values
  - Duplicates
  - Inconsistent formats
  - Irrelevant or noisy features
  - Different scales (e.g., prices vs. ages) > Normalization

We fix these issues before training the model.

`Steps in Data Preprocessing`

1. Collect the Data
   - CSV files, Databases (SQL), APIs, Web scraping, public sources (like Kaggle or UCI Repository)
2. Inspect the Data
3. Handle Missing Data
   - Remove Missing Values,
   - Fill (Impute) Missing Values
4. Handle Categorical (Text) Data
   - Label Encoding, One-Hot Encoding
5. Feature Scaling
   - Standardization (Z-score), Normalization (Min-Max Scaling)
6. Split Data
   - Training set (e.g., 80%) → model learns here,
   - Testing set (e.g., 20%) → model is evaluated here
7. Feature Selection (Optional)

##### Machine Learning Concepts

1. `Introduction to AI, ML & DL:` ML vs DL, Applications, ML pipeline, Types of ML.
2. `Data Preprocessing & Feature Engineering:` Missing data, scaling, encoding, feature selection, dimensionality reduction.
3. `Model Evaluation Techniques:` Train/Test/Validation split, k-Fold CV, confusion matrix, accuracy, F1, ROC, AUC.

##### Machine Learning Advance Topics

1. `Supervised Learning - Regression Models:` Linear Regression, Ridge, Lasso, Polynomial Regression.
2. `Supervised Learning - Classification Models:` Logistic Regression, KNN, Decision Trees, Random Forest, SVM.
3. `Introduction to Neural Networks:` Perceptron, Activation functions, Forward/Backward Propagation, Loss functions.

##### Additional Topics

![Machine Learning Workflow](/img/ml-topics.jpg)

#### Natural Language Processing (NLP) in Python

1. `NLTK (Natural Language Toolkit):` One of the oldest and most complete NLP libraries for traditional (non–deep learning) text processing.
2. `spaCy:` An industrial-strength NLP library — faster and more efficient than NLTK.
3. `Transformers (by Hugging Face):` A deep learning NLP library providing pretrained transformer models (BERT, GPT, RoBERTa, T5, etc.).
4. `Sentence Transformers:` A library built on top of Hugging Face Transformers and PyTorch for generating semantic sentence embeddings.

##### How they fit together

| `Library`                     | `Type`              | `Key Strength`    | `Typical Use`                                      |
| ----------------------------- | ------------------- | ----------------- | -------------------------------------------------- |
| `NLTK`                        | Classical NLP       | Linguistic tools  | Teaching, simple preprocessing                     |
| `spaCy`                       | Industrial NLP      | Fast pipelines    | NER, POS tagging, parsing, production NLP          |
| `Transformers (Hugging Face)` | Deep learning NLP   | Pretrained models | Summarization, question answering, text generation |
| `Sentence Transformers`       | Embedding-based NLP | Semantic meaning  | Semantic search, similarity, clustering            |

#### Neural Networks (NN)

To develop a solid understanding of `Neural Networks`, we should have a good grasp of the following key topics.

1. `Neurons & Layers:` Input, hidden, and output layers.
2. `Forward Propagation:` How data flows through the network.
3. `Backward propagation:` Gradient descent, error minimization.
4. `Activation Functions:` ReLU, Sigmoid, Tanh.
5. `Loss Functions:` MSE, Cross-Entropy.

#### Modern Generative AI and Retrieval Augmented Generation (RAG) tools.

These libraries power AI agents, retrieval systems, and diffusion models for text and images.

1. `Diffusers (by Hugging Face):` A library for generative models — especially diffusion-based image, audio, and video generation.
2. `LangChain:` A framework for building applications powered by large language models (LLMs).
3. `OpenAI (Python SDK:` Official OpenAI client library for accessing GPT, DALL·E, Whisper, etc.
4. `LlamaIndex (formerly GPT Index):` A framework for connecting LLMs to your data.
5. `FAISS (Facebook AI Similarity Search):` A vector database library for efficient similarity search.
6. `ChromaDB:` A lightweight open-source vector database optimized for use with LangChain and LlamaIndex.

##### How they fit together

| `Library`    | `Type`                | `Main Purpose`                | `Typical Use`                             |
| ------------ | --------------------- | ----------------------------- | ----------------------------------------- |
| `Diffusers`  | Generative Models     | Image / video generation      | Stable Diffusion, ControlNet              |
| `LangChain`  | LLM Framework         | Build AI agents & RAG         | Orchestrate GPT + tools                   |
| `OpenAI`     | Model API             | GPT, DALL·E, Whisper          | LLM inference and generation tasks        |
| `LlamaIndex` | RAG Framework         | Connect data to LLMs          | Document indexing & querying              |
| `FAISS`      | Vector Store (C++)    | Fast embedding search         | Large-scale vector retrieval              |
| `ChromaDB`   | Vector Store (Python) | Simple local/hosted retrieval | Lightweight RAG or semantic search setups |

#### Deep Learning (DL)

Deep Learning is a specialized subfield of machine learning that focuses on algorithms inspired by the structure and function of the human brain, known as artificial neural networks. It involves training these multi-layered networks (called deep neural networks) to automatically learn hierarchical patterns and representations from large amounts of data. To develop a solid understanding of `deep learning`, we should have a good grasp of the following key topics.

##### Deep Learning Topics

![Deep Learning Workflow](/img/dl-topics.jpg)

#### Complete Guideline: Building an ML/AI Application (Start → Deployment)

##### 1. Define the Problem Clearly > Understand what problem you’re solving.

`Steps:`
- Identify the business or real-world problem.
- Define input and output (what data you have and what you want to predict).
- Choose measurable success metrics (accuracy, RMSE, precision, etc.).
- Check if machine learning is the right solution.

`Example:`
Predict whether a customer will churn based on usage data.

##### 2. Collect and Gather Data > Obtain high-quality, relevant data.

`Sources:`
- Company databases (SQL, data warehouses)
- APIs or web scraping
- Public datasets (Kaggle, UCI, Google Dataset Search)
- IoT devices or sensors

`Tips:`
- Ensure data privacy (GDPR, HIPAA)
- Gather enough data to train the model effectively

##### 3. Data Cleaning and Preprocessing > Prepare raw data for analysis and modeling.

`Steps:`
- Handle missing values, outliers, and duplicates
- Encode categorical variables (Label/One-Hot Encoding)
- Normalize or standardize numerical features
- Split dataset:
  - `Train (70%)`
  - `Validation (15%)`
  - `Test (15%)`

`Tools:` `pandas`, `NumPy`, `scikit-learn`

##### 4. Exploratory Data Analysis (EDA) > Understand data structure and key relationships.

`Steps:`
- Visualize distributions, correlations, and trends
- Identify data patterns and anomalies
- Understand which features influence the target variable

`Tools:` `matplotlib`, `seaborn`, `plotly`

`Outcome:` Insights that guide feature selection and model choice

##### 5. Feature Engineering > Improve model performance with better features.

`Steps:`
- Create new features from existing ones
- Select important features (feature importance, correlation)
- Apply dimensionality reduction (PCA, feature selection)

`Outcome:` A clean, optimized feature set

##### 6. Model Selection and Training > Choose and train the most suitable ML algorithm.

`Steps:`
- Identify the problem type:
  - `Classification:` Logistic Regression, Random Forest, XGBoost
  - `Regression:` Linear Regression, Decision Tree, Gradient Boosting
  - `Clustering:` K-Means, DBSCAN
- Train multiple models and compare performance
- Tune hyperparameters (Grid Search, Random Search, Bayesian Optimization)

`Tools:` `scikit-learn`, `TensorFlow`, `PyTorch`, `XGBoost`, `LightGBM`

##### 7. Model Evaluation > Test model performance on unseen data.

`Common Metrics:`
- `Classification:` Accuracy, Precision, Recall, F1, ROC-AUC
- `Regression:` MAE, MSE, RMSE, R²
- `Clustering:` Silhouette Score, Davies–Bouldin Index

`Steps:`
- Evaluate on the test set
- Check for overfitting/underfitting
- Analyze errors and misclassifications

##### 8. Model Optimization and Validation > Refine model performance and ensure robustness.

`Steps:`
- Fine-tune hyperparameters
- Use ensemble techniques (bagging, boosting, stacking)
- Cross-validate results for consistency
- Retrain with optimized settings

##### 9. Model Packaging > Prepare your model for deployment.

`Steps:`
- Save trained model (`.pkl`, `.joblib`, `.pt`, `.h5`)
- Build a prediction pipeline (input → preprocess → model → output)
- Create an API for prediction using Flask or FastAPI
- Test locally

`Tools:` `Flask`, `FastAPI`, `Docker`, `Pickle`, `MLflow`

##### 10. Deployment > Make your model available for real users or systems.

`Deployment Options:`
- `Web App:` Flask/FastAPI + Streamlit/Gradio for UI
- `Cloud Platforms:` AWS Sagemaker, Google Vertex AI, Azure ML
- `Containers:` Docker or Kubernetes for scalability
- `Edge Devices:` TensorFlow Lite, ONNX for mobile or IoT

`Best Practices:`
- Use version control (`Git`)
- Automate with CI/CD pipelines
- Secure endpoints (authentication, HTTPS)

##### 11. Monitoring and Maintenance > Ensure model performance remains stable post-deployment.

`Steps:`
- Monitor prediction accuracy and latency
- Detect data drift and model degradation
- Log inputs/outputs for feedback
- Retrain periodically with new data

`Tools:` `MLflow`, `Prometheus`, `Grafana`, `Evidently AI`

##### 12. Continuous Improvement > Keep enhancing the ML system over time.

`Steps:`
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

`Problem → Data → Preparation → Modeling → Evaluation → Deployment → Monitoring → Improvement`

#### Tools & Frameworks

Build, train, and deploy ML/DL models efficiently.

| `Tool`                       | `Use`                                         |
| ---------------------------- | --------------------------------------------- |
| `Scikit-learn`               | Classical ML models                           |
| `TensorFlow / Keras`         | Deep Learning models (user-friendly)          |
| `PyTorch`                    | Deep Learning models (more control, flexible) |
| `Google Colab`               | Free GPU-powered notebooks                    |
| `Hugging Face Transformers`  | Pre-trained NLP models                        |
| `Flask / Streamlit / Docker` | Model deployment and versioning               |
| `OpenCV, YOLO`               | Computer vision applications                  |

#### Category-wise Toolset

| `Category`             | `Tools`                          |
| ---------------------- | -------------------------------- |
| `ML`                   | Scikit-learn, Pandas, Matplotlib |
| `DL`                   | TensorFlow, Keras, PyTorch       |
| `NLP`                  | spaCy, Hugging Face Transformers |
| `CV (Computer Vision)` | OpenCV, YOLO, PyTorch            |
| `Deployment`           | Flask, Streamlit, Docker         |

#### Projects

Apply your skills to real-world data and build a strong portfolio.

| `Type`                   | `Examples`                                          |
| ------------------------ | --------------------------------------------------- |
| `ML`                     | Titanic survival prediction, house price prediction |
| `DL - CV`                | Dog vs. cat classifier, facial recognition          |
| `DL - NLP`               | Sentiment analysis, chatbot, text summarizer        |
| `Reinforcement Learning` | Game-playing AI (CartPole, Atari)                   |

#### Advanced Topics

Dive deeper into specialized AI domains.

| `Topic`                             | `Focus Area`                             |
| ----------------------------------- | ---------------------------------------- |
| `Natural Language Processing (NLP)` | Transformers, BERT, GPT, text generation |
| `Computer Vision (CV)`              | Object detection, segmentation, OpenCV   |
| `Reinforcement Learning (RL)`       | Q-learning, DQN, policy gradients        |
| `MLOps`                             | Model deployment, versioning, monitoring |
| `Self-supervised Learning`          | Modern unsupervised deep learning        |
| `Graph Neural Networks (GNNs)`      | Node classification, link prediction     |

#### Recommended Platforms & Resources

- [Kaggle Python Course](https://www.kaggle.com/learn/python)
- [Google Colab](https://colab.research.google.com/)
- [Harvard CS50’s AI Course](https://cs50.harvard.edu/ai/)
- [Visual Calculus](https://mathinsight.org/calculus)
- [3Blue1Brown’s Linear Algebra Series](https://www.youtube.com/c/3blue1brown)

#### Final Tip

> Build projects, share them on GitHub, and join AI communities.
> Learning AI is a marathon, not a sprint — stay consistent and curious!

### Phase 01 Core Machine Learning (Weeks 1–4)

`Goal` Understand and implement classical ML algorithms with Python and Scikit-Learn.

`Topics`

1. Machine Learning Basics
   - What is ML? Types: supervised, unsupervised, reinforcement.
   - Data preprocessing: handling missing data, normalization, encoding.
2. Supervised Learning
   - Linear & Logistic Regression
   - Decision Trees, Random Forests
   - k-Nearest Neighbors, Naive Bayes
3. Unsupervised Learning
   - k-Means Clustering
   - Principal Component Analysis (PCA)
4. Reinforcement (Model Improvement)
   - Improving model performance
   - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Learning curves
5. Model Evaluation
   - Train/Test split
   - Cross-validation
   - Evaluation Metrics:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC–AUC
6. Practical Skills
   - Use Pandas for data manipulation
   - Use Matplotlib/Seaborn for visualization
   - Learn Scikit-Learn’s API (fit, predict, transform)

### Phase 02 Deep Learning (Weeks 5–8)

`Goal` Understand and build simple neural networks using TensorFlow or PyTorch.

`Topics`

1. Neural Network Fundamentals
   - Perceptrons, activation functions, loss functions
   - Gradient descent, backpropagation
2. Deep Learning Frameworks
   - TensorFlow / Keras basics
   - Building and training models
3. Computer Vision
   - Convolutional Neural Networks (CNNs)
   - Image classification (MNIST, CIFAR-10)
4. NLP Basics
   - Text vectorization (Bag of Words, TF-IDF)
   - Sentiment analysis using simple networks

### Phase 03 Real-World ML & Deployment (Weeks 9–12)

`Goal` Learn to handle full ML pipelines, tuning, and deployment.

`Topics`

1. Model Optimization
   - Hyperparameter tuning (GridSearchCV, RandomSearch)
   - Regularization, dropout, early stopping
2. Pipelines
   - Scikit-Learn pipelines
   - Data preprocessing automation
3. Deployment
   - Save models (joblib/pickle)
   - Deploy using Flask/FastAPI
   - Basic API integration
4. MLOps Introduction
   - Version control for data/models
   - Model monitoring basics

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


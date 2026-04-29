## More About Me – [Take a Look!](http://www.mjakaria.me)

### Roadmap of Artificial Intelligence (AI)

`Comprehensive roadmap to learn Artificial Intelligence (AI)` — from beginner to pro, structured in clear phases with recommended topics, tools, and resources.

![AI Model](/img/ai-model.png)

## OVERVIEW

AI is a broad field. The core components you'll need to master include:

- 🧮 Mathematics for Machine Learning
- 💻 Programming in Python
- 🤖 Machine Learning (ML)
- 🧠 Neural Networks (NN)
- 🔍 Deep Learning (DL)
- 🧰 Tools & Frameworks
- 💼 Projects
- 🚀 Advanced Topics (NLP, Computer Vision, Reinforcement Learning)

### Mathematics for Machine Learning

1. `Linear Algebra:` Vectors, matrices, dot product, eigenvalues.
2. `Probability & Statistics:` Distributions, Bayes’ Theorem, expectation.
3. `Calculus Basics:` Derivatives, gradients, chain rule.
4. `Optimization:` Gradient descent, cost functions

### Programming in Python-[Visit](https://github.com/jakir-ruet/mastering-with-python)

Be comfortable writing and debugging Python code for data and ML.

1. Python Syntax, Python Comments, Python Variables, Python Data Types, Python Numbers, Python Casting, Python Strings, Python Booleans, Python Operators.
2. Python Lists, Python Tuple, Python Sets, Python Dictionaries Python If...Else, Python Match.
3. Python While Loops, Python for Loops, Python Functions, Python Lambda.
4. Python Classes/Objects, Python Inheritance, Python Iterators Python Polymorphism.
5. Python Polymorphism, Python Modules Python Dates, Python Math.
6. Python JSON, Python RegEx, Python PIP, Python Try...Except.
7. Python String Formatting, Python User Input, Python VirtualEnv.
8. Python File Handling, Python Read Files, Python Write/Create Files Python Delete Files.

### Python Library (Programming for ML and DL)

1. `Matplotlib:` (creating static, animated, and interactive plots).
2. `Seaborn:` (high-level visualization library)
3. `SciPy:` (scientific computing library built on NumPy)
4. `Pandas:` (Data Frames Handles labeled data)
5. `NumPy:` Handles numerical data (arrays, matrices).

### Machine Learning (ML)

Machine Learning is a branch of artificial intelligence (AI) that enables computers to learn patterns and make decisions or predictions from data without being explicitly programmed. Instead of following fixed rules, machine learning algorithms automatically improve their performance over time as they are exposed to more data and experience.

In Simple terms

| Traditional programming                                           | Machine Learning                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| You give the computer `rules` + `data` → it produces an `answer`. | You give the computer `data` + `answers` → it learns the `rules`. |

### How Machine Learning Works (at a high level)

1. Collect Data > (e.g., customer info, images, text)
2. Prepare Data > Clean, normalize, and split into training/testing sets.
3. Choose a Model > (e.g., linear regression, decision tree, neural network)
4. Train the Model > Feed it data so it learns patterns.
5. Test the Model > Evaluate accuracy on new (unseen) data.
6. Use the Model > Make predictions or decisions automatically.

### Main Types of Machine Learning

| Type                       | What it does                                    | Example                               | Student Analogy                     | Why                                                  |
| -------------------------- | ----------------------------------------------- | ------------------------------------- | ----------------------------------- | ---------------------------------------------------- |
| **Supervised Learning**    | Learns from labeled data (has correct answers)  | Predict house prices, spam detection  | Student learns with an answer sheet | Can compare answers and improve using correct labels |
| **Unsupervised Learning**  | Finds hidden patterns in unlabeled data         | Customer segmentation, topic grouping | Student has no answer sheet         | Must discover patterns on their own                  |
| **Reinforcement Learning** | Learns by reward and punishment (trial & error) | Game AI, robots, self-driving cars    | Student learns from marks/rewards   | Improves actions based on feedback                   |

> Simple memory trick
>
> - `Supervised` → Teacher shows answers
> - `Unsupervised` → Student discovers patterns
> - `Reinforcement` → Learn by reward & punishment

## Large Language Model - `LLM`

An LLM is a computer model that learns language patterns from huge text datasets and can read, write, explain, translate, and answer questions like a human. Examples of real-world LLM families:
GPT models, LLaMA, Claude, Mistral, Gemini etc.

> Key Characteristics
>
> - `Large` → trained on billions or trillions of words
> - `Language` → works with human text (English, Bangla, etc.)
> - `Model` → a mathematical neural network that learns patterns

### LLM Workflow (Real System)

| Step                           | What Happens                                        | Example                                                                       |
| ------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| **1. User Prompt**             | You enter text                                      | `Explain Docker in simple terms.`                                             |
| **2. Tokenization**            | Text → tokens → numbers                             | `[Explain, Docker, in, simple, terms,.]` → `[1245, 9821, 304, 762, 4501, 13]` |
| **3. Transformer Processing**  | Tokens → vectors → attention layers analyze context | `simple` influences explanation style                                         |
| **4. Probability Calculation** | Model computes probability for next token           | `P(Docker)=0.30`, `P(It)=0.42`, `P(A)=0.10`                                   |
| **5. Next Token Prediction**   | Highest/sampled probability selected                | `Docker`                                                                      |
| **6. Generated Response**      | Repeats token-by-token until complete               | `Docker is a platform that...`                                                |

> `User Prompt → Tokenization → Transformer Processing → Probability Calculation → Next Token Prediction → Generated Response`

### Prompt

A prompt is the input (instruction, question, or text) that you give to an AI model to generate a response.

> Simply: `Prompt` = `What you tell the AI`

#### Prompt Types

| Type                 | What It Means                  | Example                                               |
| -------------------- | ------------------------------ | ----------------------------------------------------- |
| **Instruction**      | Direct command                 | `Explain Docker.`                                     |
| **Question**         | Asking something               | `What is Kubernetes?`                                 |
| **Zero-Shot**        | Task without examples          | `Translate Hello to Spanish.`                         |
| **One-Shot**         | One example given              | `Hi → Hola`<br>`Thanks →`                             |
| **Few-Shot**         | Multiple examples given        | `Bad → Negative`<br>`Great → Positive`<br>`Awesome →` |
| **Role-Based**       | Assign a role                  | `Act as a DevOps engineer.`                           |
| **Chain-of-Thought** | Ask for step-by-step reasoning | `Solve 25 × 12 step by step.`                         |

## Big Picture

Machine Learning is NOT just algorithms. It is a complete engineering system:

1. Data Engineering
2. Feature Engineering
3. Modeling
4. Evaluation
5. Optimization
6. Deployment
7. Monitoring

### Core ML Pipeline (End-to-End System)

| Steps | Stage                    | What Happens               | Purpose              | Input → Output           | Tools                           | Real Example (Dog Detection)    |
| :---: | ------------------------ | -------------------------- | -------------------- | ------------------------ | ------------------------------- | ------------------------------- |
|   1   | Problem Definition       | Define the ML problem      | Understand goal      | Business idea → ML task  | Domain knowledge                | `Is this image a dog or not?`   |
|   2   | Data Collection          | Gather raw data            | Build dataset        | Images/CSV/API → Dataset | APIs, SQL, Kaggle, Web scraping | Dog + non-dog images            |
|   3   | Data Understanding (EDA) | Analyze data patterns      | Understand structure | Raw data → Insights      | Pandas, Matplotlib, Seaborn     | Check image size, labels        |
|   4   | Data Preprocessing       | Clean & prepare data       | Fix data issues      | Raw data → Clean data    | Pandas, NumPy, Scikit-learn     | Resize images, normalize pixels |
|   5   | Feature Engineering      | Convert data into features | Improve model input  | Clean data → Features    | PCA, Encoding, TF-IDF           | Image → pixel vectors           |
|   6   | Train/Test Split         | Split dataset              | Avoid overfitting    | Dataset → Train + Test   | sklearn.model_selection         | 80% train, 20% test             |
|   7   | Model Selection          | Choose algorithm           | Find best model      | Features → Model         | SVM, RF, KNN, XGBoost           | Random Forest chosen            |
|   8   | Model Training           | Learn patterns             | Build intelligence   | Train data → Model       | fit() (Sklearn, PyTorch)        | Model learns dog patterns       |
|   9   | Evaluation               | Measure performance        | Check accuracy       | Predictions → Metrics    | Accuracy, F1, ROC-AUC           | 95% accuracy                    |
|  10   | Hyperparameter Tuning    | Improve model              | Optimize performance | Model → Better model     | GridSearchCV, Optuna            | Improve Random Forest           |
|  11   | Packaging                | Save model                 | Reuse model          | Model → File             | Pickle, Joblib                  | model.pkl saved                 |
|  12   | Deployment               | Make model live            | Real-world usage     | Model → API/App          | Flask, FastAPI, Docker          | Dog detection web app           |
|  13   | Monitoring               | Track performance          | Maintain model       | Logs → Metrics           | MLflow, Grafana                 | Detect performance drop         |
|  14   | Continuous Improvement   | Retrain model              | Keep improving       | New data → Updated model | CI/CD pipelines                 | Better dog detection            |

> - Machine Learning ecosystem = `Data (Pandas) → Model (Sklearn/XGBoost/PyTorch) → Optimize (Optuna) → Save (Joblib/Pickle) → Deploy (Flask/Docker)`
> - EDA = Exploratory Data Analysis

### Preprocessing & Feature Engineering Stack

| Category         | Techniques                      | Purpose                | Tools       |
| ---------------- | ------------------------------- | ---------------------- | ----------- |
| Encoding         | Label, One-Hot, Ordinal, Binary | Convert text → numbers | Sklearn     |
| Scaling          | Standard, MinMax, Robust        | Normalize data         | Sklearn     |
| Transformation   | Log, Power                      | Fix skewness           | NumPy       |
| Text Processing  | TF-IDF, Tokenization, BERT      | Convert text → vectors | NLTK, SpaCy |
| Dim Reduction    | PCA, Feature selection          | Reduce features        | Sklearn     |
| Imbalance Fix    | SMOTE, Under/Over sampling      | Balance dataset        | Imblearn    |
| Feature Creation | Aggregation, math features      | Improve accuracy       | Pandas      |

### Machine Learning Models

| Type           | Algorithms                            | Use Case           |
| -------------- | ------------------------------------- | ------------------ |
| Regression     | Linear, Ridge, Lasso, SVR, XGBoost    | Predict numbers    |
| Classification | Logistic, SVM, Random Forest, XGBoost | Predict categories |
| Clustering     | K-Means, DBSCAN                       | Group data         |
| Ensemble       | Bagging, Boosting                     | Improve accuracy   |

### Evaluation System

| Task           | Metrics                                  | Meaning                |
| -------------- | ---------------------------------------- | ---------------------- |
| Classification | Accuracy, Precision, Recall, F1, ROC-AUC | Classification quality |
| Regression     | MAE, MSE, RMSE, R²                       | Prediction error       |
| Clustering     | Silhouette Score                         | Group quality          |

### Model Tuning System

| Method                | Purpose              |
| --------------------- | -------------------- |
| GridSearchCV          | Try all combinations |
| RandomSearch          | Faster tuning        |
| Optuna                | Smart optimization   |
| Bayesian Optimization | Advanced tuning      |

### Deep Learning Core

| Concept              | Meaning                        |
| -------------------- | ------------------------------ |
| Neural Networks      | Brain-like model               |
| Forward Propagation  | Input → Output flow            |
| Backpropagation      | Learning process               |
| Activation Functions | Decision rules (ReLU, Sigmoid) |
| Loss Function        | Error measurement              |
| Optimizer            | Learning improvement           |

### Computer Vision (CV)

| Area              | Tools               | Example              |
| ----------------- | ------------------- | -------------------- |
| CNN               | TensorFlow, PyTorch | Image classification |
| Pretrained Models | ResNet, VGG         | Transfer learning    |
| Image Processing  | OpenCV              | Face detection       |

### NLP Ecosystem

| Library               | Role       | Use Case         |
| --------------------- | ---------- | ---------------- |
| NLTK                  | Basic NLP  | Tokenization     |
| spaCy                 | Fast NLP   | NER, POS tagging |
| Transformers          | Deep NLP   | GPT, BERT        |
| Sentence Transformers | Embeddings | Semantic search  |

> NLP = Natural Language Processing
> NLTK = Natural Language Toolkit
> GPT = Generative Pre-trained Transformer
> BERT = Bidirectional Encoder Representations from Transformers
> POS = Part-of-Speech (Tagging)

### Generative AI & Retrieval Augmented Generation (RAG) Ecosystem

| Tool       | Purpose               |
| ---------- | --------------------- |
| OpenAI API | GPT models            |
| LangChain  | AI agent framework    |
| LlamaIndex | Data + LLM connection |
| FAISS      | Vector search         |
| ChromaDB   | Vector database       |
| Diffusers  | Image generation      |

### Machine Learning & AI Framework Ecosystem

| Category   | Tools                              | Purpose                                                      |
| ---------- | ---------------------------------- | ------------------------------------------------------------ |
| ML         | Scikit-learn, XGBoost, LightGBM    | Classical ML models (classification, regression, clustering) |
| DL         | TensorFlow, Keras, PyTorch, FastAI | Neural networks, deep learning models                        |
| NLP        | SpaCy, NLTK, Transformers          | Text processing, language models (BERT, GPT)                 |
| Deployment | Flask, FastAPI, Docker             | Model API, serving, containerization                         |
| Data       | Pandas, NumPy                      | Data manipulation and numerical computing                    |

### Advanced ML & DL Tools (Expanded)

| Tool                | Type              | Purpose                                                        |
| ------------------- | ----------------- | -------------------------------------------------------------- |
| XGBoost             | ML (Boosting)     | High-performance gradient boosting for tabular data            |
| LightGBM            | ML (Boosting)     | Faster, memory-efficient gradient boosting (Microsoft)         |
| Joblib              | Model Persistence | Save/load ML models efficiently (large arrays, sklearn models) |
| Pickle              | Serialization     | Save/load Python objects (general-purpose)                     |
| Optuna              | Optimization      | Automatic hyperparameter tuning                                |
| TensorFlow/Keras    | Deep Learning     | Neural networks, production-ready DL framework                 |
| PyTorch/Torchvision | Deep Learning     | Research-friendly deep learning + computer vision              |
| FastAI              | High-level DL API | Simplified deep learning built on PyTorch                      |

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

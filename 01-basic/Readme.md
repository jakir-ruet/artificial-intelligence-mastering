### Machine Learning (ML)

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed. Instead of following strict rules coded by a programmer, ML systems improve their performance as they are exposed to more data.

> Key points:

- Learns from data.
- Improves automatically over time.
- Can make predictions, classifications, or decisions.

#### Machine Learning Classification at a glance

1. `Supervised Learning`
   - Regression
   - Classification
2. `Unsupervised Learning`
   - Clustering
   - Dimensionality Reduction
   - Association Learning
3. `Reinforcement Learning`
   - Value-Based
   - Policy-Based
   - Actor-Critic
4. `Semi-Supervised Learning`
5. `Self-Supervised Learning`
6. `Deep Learning`
7. `Discriminative vs Generative Models`

#### Brief Discussion

1. `Supervised Learning` > Learns from labeled data (input-output pairs). Predict the output for new inputs.

- `Regression` (predict continuous values)
  - `Examples:` Linear Regression, Polynomial Regression, Support Vector Regression
  - `Use Case:` Predict house prices, stock prices

- `Classification` (predict categories/classes)
  - `Examples:` Logistic Regression, Decision Tree, Random Forest, SVM, Neural Networks
  - `Use Case:` Email spam detection, image recognition

2. `Unsupervised Learning` > Learns from unlabeled data; finds patterns or structures. Discover hidden patterns, clusters, or structure in data.

- `Clustering`
  - `Examples:` K-Means, Hierarchical Clustering, DBSCAN
  - `Use Case:` Customer segmentation

- `Dimensionality Reduction`
  - `Examples:` PCA, t-SNE, Autoencoders
  - `Use Case:` Reduce feature space, visualization

- `Association Learning`
  - `Examples:` Apriori, FP-Growth
  - `Use Case:` Market basket analysis, recommendation systems

3. `Reinforcement Learning (RL)` > Learns by interacting with the environment and receiving rewards or penalties.
`Goal:` Maximize cumulative reward.

- `Value-Based Methods`
  - `Examples:` Q-Learning, Deep Q Networks (DQN)

- `Policy-Based Methods`
  - `Examples:` Policy Gradient, REINFORCE

- `Actor-Critic Methods`
  - `Examples:` A3C, PPO

`Use Case:` Game AI, robotics, self-driving cars

4. Semi-Supervised Learning
`Definition:` Uses a small amount of labeled data and a large amount of unlabeled data.
`Use Case:` Medical image classification, speech recognition

5. Self-Supervised Learning
`Definition:` Labels are generated automatically from the data itself.
`Use Case:` Natural Language Processing (e.g., GPT, BERT), computer vision

6. Deep Learning (subset of ML)
`Definition:` Uses multi-layered neural networks to model complex patterns.
`Use Case:` Image recognition, NLP, speech recognition, autonomous driving

`Examples:`
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN, LSTM)
- Transformers (BERT, GPT)
- Generative Models (GANs, VAEs)

7. Discriminative vs Generative Models
- `Discriminative Models`
  - Learn \( P(Y|X) \), good for classification
  - `Examples:` Logistic Regression, SVM, Neural Networks

- `Generative Models`
  - Learn \( P(X|Y) \) and \( P(Y) \), can generate new data
  - `Examples:` Naive Bayes, GANs, VAEs, HMM

### Machine Learning Workflow Summary

1. `Pre-Preprocessing and Feature Engineering`
   - Encoding and Binning
     - Label Encoding
     - One Hot Encoding
     - Ordinal Encoding
     - Binary Encoding
     - Frequency Encoding
   - Scaling and Transformation
     - Standard Scalar
     - MinMax Scalar
     - Robust Scalar
     - MaxAbs Scalar
     - Log Transformation
     - Power Transformation
   - Text Processing
     - Regex
     - Stemming/Lemmatization
     - Tokenization
     - POS Tagging
     - Stopwords Removal
     - Vectorization
       - Count
       - Tf-Idf
       - Word2Vec
       - BERT Embeddings
   - Dim Reduction
       - PCA
       - Filter Methods
       - Embedded Methods
       - Wrapper Methods
   - Feature Creation
      - Mathematical Combinations
      - Integration Features
      - Aggregations
   - Imbalanced Dataset
      - Under Sampling
      - Over Sampling
      - SMOT Tomek
   - Python Libraries
      - Pandas
      - NumPy
      - NLTK/SpaCy for Text
      - Imblearn
      - Statsmodels
      - Scikit-learn
2. `Model Building`
   - Supervised Learning
      - Regression
        - Linear Regression
        - Polynomial
        - Lasso/Ridge
        - ElasticNet
        - SVR
        - KNN Regressor
        - Booting Regressor
        - Random Forest Regressor
      - Classification
        - Tree Based Models
          - Decision Tree
          - Random Forest
          - Extra Trees
          - Gradient Boosting Machine (GBM)
          - XGBoost
          - AdaBoost
          - LightGBM
          - CatBoost
        - KNN Classifier
        - Logistic Regression
        - SVC
        - Naive Bays
        - Ensemble Techniques
   - Unsupervised Learning
     - Clustering
     - PCA
   - Model Evaluation
     - Bias Variance Tradeoff
     - Cross Validation
       - K-Fold
       - Stratified k-Fold
       - LOOCV
       - Time Series Split
     - Confusion Matrix for Classification
       - Accuracy
       - Precision
       - Recall
       - F1 Score
       - AUC/ROC
     - Model Evaluation for Regression
       - R-Squared (Accuracy)
       - MSE
       - MAE
       - RMSE
   - Model Tuning
     - Grid Search
     - Random Search
     - Genetic Algorithm
     - Bayesian Optimization

### Deep Learning Workflow Summary

1. DL Basics
   - ANN Architecture
   - Neural Networks Architecture
   - Neural, Input-Output, Hidden Layers
   - Activation Functions (ReLU, Sigmoid, SoftMax, Swish, Tanh & Linear)
   - Loss Functions (Cross-Entropy, MSE/MAE)
   - Optimizers (SGD, Adam, RMSprop)
   - Forward and Backward Propagation
   - Epochs, Batch Size, Learning Rate
   - Overfitting and Regularization (Dropout, L2)
   - Vanishing and Exploring Gradient
2. Image/Video (CV)
   - Image Processing
      - Normalization
      - Augmentation
      - Color Conversion
   - Image Classification using CNN
      - Raw Pixels
      - CNNs (Backbone)
      - Pretrained CNNs
        - VGG
        - ResNet
        - ImageNet
        - Others
3. Text (NLP)
   - Text Processing (Details in ML)
   - Sequence Models
    - RNNs
    - LSTMs
    - GRUs
    - BERT
   - Core Text Tasks in NLP
    - Text Classification
    - Sentiment Analysis
4. Library/Framework
   - Keras
   - TensorFlow

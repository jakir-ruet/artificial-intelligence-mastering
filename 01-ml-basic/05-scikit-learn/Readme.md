### Scikit-Learn (sklearn)

It's one of the most popular Python libraries for Machine Learning.

It provides simple tools to:

- Build ML models (classification, regression, clustering)
- Preprocess data (cleaning, scaling, encoding)
- Split datasets (train/test)
- Evaluate model performance

> It is built on top of `NumPy`, `SciPy`, and `Matplotlib`, so it integrates well with data science workflows.

#### Why Scikit-Learn is popular?

- Easy to use (clean API)
- Fast to prototype ML models
- Huge collection of algorithms
- Great documentation
- Industry standard for classical ML

#### Main Types of Machine Learning & Models

| Type                       | What it does                                    | Example                               | Student Analogy                     | Why                                                  |
| -------------------------- | ----------------------------------------------- | ------------------------------------- | ----------------------------------- | ---------------------------------------------------- |
| **Supervised Learning**    | Learns from labeled data (has correct answers)  | Predict house prices, spam detection  | Student learns with an answer sheet | Can compare answers and improve using correct labels |
| **Unsupervised Learning**  | Finds hidden patterns in unlabeled data         | Customer segmentation, topic grouping | Student has no answer sheet         | Must discover patterns on their own                  |
| **Reinforcement Learning** | Learns by reward and punishment (trial & error) | Game AI, robots, self-driving cars    | Student learns from marks/rewards   | Improves actions based on feedback                   |

> **Simple memory trick**
>
> - `Supervised` → Teacher shows answers
> - `Unsupervised` → Student discovers patterns
> - `Reinforcement` → Learn by reward & punishment
>
```bash
Machine Learning
│
├── Supervised Learning
│     ├── Classification Models
│     │     ├── Binary (Spam detection)
│     │     ├── Multi-class (Animal classification)
│     │     └── Multi-label (Image tagging)
│     │
│     └── Regression Models
│           ├── Linear (House price)
│           ├── Polynomial (Growth curve)
│           └── Regularized (Salary prediction)
│
├── Unsupervised Learning
│     ├── Clustering Models
│     │     ├── K-Means (Customer groups)
│     │     ├── DBSCAN (Fraud detection)
│     │     └── Hierarchical (Document grouping)
│     │
│     └── Dimensionality Reduction
│           ├── PCA (Feature reduction)
│           ├── t-SNE (Visualization)
│           └── SVD (Text processing)
│
├── Semi-Supervised Learning
│     ├── Label Propagation (Image labeling)
│     └── Self-training (Text classification)
│
└── Reinforcement Learning
      ├── Value-based (Q-learning)
      ├── Policy-based (Robot control)
      └── Model-based (Self-driving cars)
```

> **Master Summary**
>
> - Classification → Predict categories
> - Regression → Predict numbers
> - Clustering → Group data
> - Dimensionality Reduction → Compress data
> - Reinforcement Learning → Learn by reward

### Artificial Intelligence

Answer: Artificial Intelligence is a branch of computer science focused on creating systems that can perform tasks requiring human-like intelligence, such as learning, reasoning, perception, language understanding, and decision-making.

### What is the difference between AI, ML, DL, and Generative AI?

AI is the broad field of intelligent systems. Machine Learning is a subset of AI where systems learn patterns from data. Deep Learning is a subset of ML based on multi-layer neural networks. Generative AI focuses on creating new content such as text, images, audio, video, or code.

### AI is commonly classified by capability into three types

**1. ANI — Artificial Narrow Intelligence** Artificial Narrow Intelligence performs a specific task or limited set of tasks.

**2. AGI — Artificial General Intelligence** Artificial General Intelligence refers to AI capable of performing a broad range of intellectual tasks at roughly human-level generality.

**3. ASI — Artificial Super Intelligence** Artificial Super Intelligence refers to hypothetical AI that exceeds human intelligence across most or all major cognitive domains.

| Type    | Full Name                       | Meaning                        | Example                           |
| ------- | ------------------------------- | ------------------------------ | --------------------------------- |
| **ANI** | Artificial Narrow Intelligence  | Specialized for specific tasks | Chatbot, fraud detection          |
| **AGI** | Artificial General Intelligence | Broad human-level capability   | No established real-world example |
| **ASI** | Artificial Superintelligence    | Intelligence beyond humans     | Hypothetical                      |

![AI Model](/img/ai-fundamentals/ani-agi-asi.jpeg)

### AI vs ML vs DL vs Generative AI

```bash
Artificial Intelligence (AI)
└── Machine Learning (ML)
    └── Deep Learning (DL)
        └── Many modern Generative AI systems
```

| Concept           | Meaning                              | Example                       |
| ----------------- | ------------------------------------ | ----------------------------- |
| **AI**            | Broad field of intelligent systems   | Rule-based expert system      |
| **ML**            | Learns patterns from data            | Dropout prediction            |
| **DL**            | ML using multi-layer neural networks | Image recognition             |
| **Generative AI** | Generates new content                | Text, image, audio generation |

ML is a subset of AI. DL is a subset of ML. Generative AI is a capability/category of AI, and many modern GenAI systems are built using Deep Learning and Transformers.

![AI Model](/img/ai-fundamentals/ai-model.png)

### Main Types of Machine Learning & Models

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

### Core AI Terminology

| Term          | Meaning                     | Student Example            |
| ------------- | --------------------------- | -------------------------- |
| **Dataset**   | Complete collection of data | 10,000 student records     |
| **Sample**    | One record                  | One student                |
| **Feature**   | Input variable              | Attendance                 |
| **Label**     | Expected output             | Dropout = Yes              |
| **Algorithm** | Learning method             | Logistic Regression        |
| **Model**     | Learned system              | Trained dropout predictor  |
| **Training**  | Learning process            | Learn from historical data |
| **Inference** | Using model                 | Predict new student risk   |

#### 1. Dataset - A dataset is the complete collection of data used in an AI/ML system.

| STUDENT_ID | ATTENDANCE | MARKS | FAILED | DROPOUT |
| ---------- | ---------- | ----- | ------ | ------- |
| 101        | 95         | 88    | 0      | NO      |
| 102        | 60         | 45    | 2      | YES     |
| 103        | 78         | 67    | 1      | NO      |

> This entire table is the dataset.

#### 2. Sample - One individual record is a sample.

**STUDENT_ID** Student 102

| Title      | Sign | Marks |
| ---------- | ---- | ----- |
| Attendance | =    | 60    |
| Marks      | =    | 45    |
| Failed     | =    | 2     |
| Dropout    | =    | YES   |

#### 3. Feature - A feature is an input variable used by the model.

Features:
- Attendance
- Marks
- Failed Subjects
- Fee Delay

#### 4. Label - A label is the expected output or target.

```bash
Features:
Attendance = 60
Marks = 45
Failed = 2

Label:
Dropout = YES
```

#### 5. Algorithm - An algorithm is the learning procedure.

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest

#### 6. Model - A model is the result produced after an algorithm learns from data.

```bash
Algorithm + Training Data
          ↓
       Training
          ↓
     Trained Model
```

#### 7. Training - Training is the learning process.

```bash
Historical Data
      ↓
Algorithm
      ↓
Adjust Parameters
      ↓
Reduce Errors
      ↓
Trained Model
```

**Example:**

1. Student 1: Attendance 95%, Marks 88 → NO DROPOUT
2. Student 2: Attendance 55%, Marks 40 → DROPOUT
3. Student 3: Attendance 70%, Marks 60 → NO DROPOUT

> The model learns patterns from known examples.

#### 8. Inference - Inference means using a trained model on new data.

```bash
New Student

Attendance = 58
Marks = 42
Failed = 3
      ↓
Trained Model
      ↓
Dropout Risk = 89%
```

> No new learning is required here. The existing model is being used for prediction.

### Training vs Inference - Simple Comparison

| Training               | Inference                       |
| ---------------------- | ------------------------------- |
| Model learns from data | Uses learned patterns           |
| Learns patterns        | Trained model makes predictions |
| Historical data        | New data                        |
| Expensive              | Usually cheaper                 |
| Slower                 | Usually faster                  |
| Updates parameters     | Parameters normally fixed       |
| Done periodically      | Can happen continuously         |

### Complete Flow

```bash
Dataset
   ↓
Features (X) + Labels (y)
   ↓
Algorithm
   ↓
Training
   ↓
Model
   ↓
New Data
   ↓
Inference
   ↓
Prediction
```

### What is the difference between an algorithm and a model?

An algorithm is the method used to learn patterns from data, while a model is the trained artifact produced after applying that algorithm to training data.

### AI Problem Types —

**1. Classification** - Classification predicts a category or class.
**2. Regression** - Regression Analysis predicts a continuous numeric value.
**3. Clustering** - Cluster Analysis automatically groups similar data.
**4. Anomaly Detection** - Anomaly Detection finds unusual or abnormal patterns.

| Problem Type          | Output       | Example                |
| --------------------- | ------------ | ---------------------- |
| **Classification**    | Category     | Dropout: Yes / No      |
| **Regression**        | Number       | Final score: 78.5      |
| **Clustering**        | Groups       | Group similar students |
| **Anomaly Detection** | Unusual case | Suspicious attendance  |

| Requirement                  | Problem Type      |
| ---------------------------- | ----------------- |
| Will student drop out?       | Classification    |
| What will final score be?    | Regression        |
| Group similar students       | Clustering        |
| Detect suspicious attendance | Anomaly Detection |

### Model Evaluation - How good is the model?

| Metric    | Full Form                                            | Meaning                                                     | When to Use                         |
| --------- | ---------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------- |
| Accuracy  | Accuracy Score                                       | Percentage of correct predictions                           | When dataset is balanced            |
| Precision | Precision Score                                      | Correct positive predictions out of all predicted positives | When false positives are costly     |
| Recall    | Recall (Sensitivity)                                 | Correct positives out of all actual positives               | When missing positives is costly    |
| F1 Score  | F1 Score                                             | Harmonic mean of Precision and Recall                       | For imbalanced datasets             |
| ROC-AUC   | Receiver Operating Characteristic - Area Under Curve | Measures model’s ability to distinguish between classes     | Binary classification problems      |
| Log Loss  | Logarithmic Loss                                     | Penalizes confident wrong predictions                       | Probabilistic classification models |

#### Accuracy Calculation

`Accuracy` = `(𝑇𝑃 + 𝑇𝑁)`/`(𝑇𝑃 + 𝑇𝑁 + 𝐹𝑃 + 𝐹𝑁)`

Where,
- `TP:` True Positive (correctly predicted `yes`)
- `TN:` True Negative (correctly predicted `no`)
- `FP:` False Positive (incorrectly predicted `yes`)
- `FN:` False Negative (missed a `yes`)

> Works well when classes are balanced. Misleading for imbalanced data (e.g., 99% non-fraud, 1% fraud).

#### Precision Calculation

`Precision` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑃)`

> High precision = few false alarms.

#### Recall Calculation

`Recall` = `𝑇𝑃`/`(𝑇𝑃 + 𝐹𝑁)`

> High recall = you catch most of the positive cases.

#### F1 Score Calculation

`𝐹1` = `2` × (`(Precision × Recall)`/`(Precision + Recall)`)

> Useful when you want a balance between precision and recall.

#### Predicting Machine Maintenance

| Strategy               | Precision | Recall   | Note                                 |
| ---------------------- | --------- | -------- | ------------------------------------ |
| Predict all machines   | Low       | High     | Catch all potential issues           |
| Predict only when sure | High      | Low      | Avoid unnecessary maintenance        |
| F1 score               | Balanced  | Balanced | Trade-off between precision & recall |

> High accuracy does not always mean a good model.

**1. Underfitting** - Underfitting happens when the model does not learn enough.

- Training Accuracy = 60%
- Test Accuracy     = 58%

**2. Good Fit**- The model learns real patterns and generalizes well.

- Training Accuracy = 90%
- Test Accuracy     = 87%

**3. Overfitting** Overfitting happens when the model learns training data too specifically.

- Training Accuracy = 99%
- Test Accuracy     = 70%

> Key Takeaway
> - Underfitting → Model learns too little
> - Good Fit → Model learns useful patterns
> - Overfitting → Model learns training data too specifically

### AI/ML Lifecycle

| Step                       | Purpose                   | Student Example          |
| -------------------------- | ------------------------- | ------------------------ |
| **1. Problem Definition**  | Define business goal      | Predict student dropout  |
| **2. Data Collection**     | Gather relevant data      | Attendance, marks, fees  |
| **3. Data Preparation**    | Clean and transform data  | Handle missing marks     |
| **4. Feature Engineering** | Create useful inputs      | Attendance rate          |
| **5. Model Training**      | Learn patterns            | Train classifier         |
| **6. Evaluation**          | Measure quality           | Precision, recall, F1    |
| **7. Deployment**          | Serve predictions         | REST API                 |
| **8. Monitoring**          | Track production behavior | Accuracy, drift, latency |
| **9. Retraining**          | Update with new data      | Train new model version  |


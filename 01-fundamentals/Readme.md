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

#### Dataset - A dataset is the complete collection of data used in an AI/ML system.

| STUDENT_ID | ATTENDANCE | MARKS | FAILED | DROPOUT |
| ---------- | ---------- | ----- | ------ | ------- |
| 101        | 95         | 88    | 0      | NO      |
| 102        | 60         | 45    | 2      | YES     |
| 103        | 78         | 67    | 1      | NO      |

> This entire table is the dataset.

#### Sample - One individual record is a sample.

**STUDENT_ID** Student 102

| Title      | Sign | Marks |
| ---------- | ---- | ---- |
| Attendance | =    | 60   |
| Marks      | =    | 45   |
| Failed     | =    | 2    |
| Dropout    | =    | YES  |

#### Feature - A feature is an input variable used by the model.

Features:
- Attendance
- Marks
- Failed Subjects
- Fee Delay

#### Label - A label is the expected output or target.

```bash
Features:
Attendance = 60
Marks = 45
Failed = 2

Label:
Dropout = YES
```

#### Algorithm - An algorithm is the learning procedure.

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest

#### Model - A model is the result produced after an algorithm learns from data.

```bash
Algorithm + Training Data
          ↓
       Training
          ↓
     Trained Model
```

#### Training - Training is the learning process.

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

> The model learns patterns from known examples.

#### Inference - Inference means using a trained model on new data.

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

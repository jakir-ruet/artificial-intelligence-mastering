### Transformer

It's an attention-based neural network architecture composed of embedding layers, positional encoding, multi-head self-attention, feed-forward neural networks, residual connections, and layer normalization. It is the foundation of modern Large Language Models (LLMs).

> Transformer = Attention-based neural network architecture for understanding relationships between tokens.

**Key Characteristics**

| Feature            | Description                          |
| ------------------ | ------------------------------------ |
| **Architecture**   | Deep Neural Network                  |
| **Core Mechanism** | Self-Attention                       |
| **Input**          | Sequence (text, code, DNA, etc.)     |
| **Processing**     | Parallel                             |
| **Memory**         | Learns long-range dependencies       |
| **Foundation Of**  | GPT, BERT, T5, Llama, Gemini, Claude |

#### RNN vs LSTM vs CNN

| Model    | Full Name                    | Best For        | Core Idea                                                                             |
| -------- | ---------------------------- | --------------- | ------------------------------------------------------------------------------------- |
| **CNN**  | Convolutional Neural Network | Images, videos  | Learns **spatial patterns** (edges, shapes, objects)                                  |
| **RNN**  | Recurrent Neural Network     | Sequential data | Learns from **previous time steps** using a hidden state                              |
| **LSTM** | Long Short-Term Memory       | Long sequences  | Uses **memory cells and gates** to retain important information over longer sequences |

**1. CNN (Convolutional Neural Network)**  is a deep learning architecture that learns spatial (dimensional, geographical, locational, and spacial) features from grid-like data using convolutional filters, making it highly effective for image processing tasks.

1. Input `Image`
2. Processing

```bash
Image
   ↓
Convolution
   ↓
Feature Maps
   ↓
Pooling
   ↓
Classifier
```

**Example**

```bash
Student ID Card Image
        ↓
		 CNN
        ↓
Recognize Student
```

**Common Applications**

- Image Classification
- Face Recognition
- Object Detection
- Medical Imaging
- OCR
- Self-driving Cars

**2. RNN (Recurrent Neural Network)** is a neural network architecture designed for sequential data, where each output depends on the current input and the previous hidden state.

1. Input `Sequence`
2. Processing `Day1 → Day2 → Day3 → Day4`

> Each step remembers information from the previous step.

**Example**

```bash
Attendance
90%
85%
70%
60%
↓
Predict Dropout
```

**Common Applications**

1. Time Series Forecasting
2. Speech Recognition
3. Language Modeling
4. Sequential Classification

**3. LSTM (Long Short-Term Memory)** is a specialized RNN that uses memory cells and gating mechanisms to preserve relevant information across long sequences, helping mitigate the vanishing-gradient problem.

1. Processing

```bash
Input
    ↓
Forget Gate
    ↓
Input Gate
    ↓
Cell State
    ↓
Output Gate
```

**Example**

```bash
Semester 1 GPA
Semester 2 GPA
Semester 3 GPA
Semester 4 GPA
↓
Predict Graduation Risk
```

**Common Applications**

1. Machine Translation
2. Speech Recognition
3. Time Series Prediction
4. Stock Prediction
5. Long Sequence Modeling

### Key Differences

| Feature             | CNN         | RNN                    | LSTM                   |
| ------------------- | ----------- | ---------------------- | ---------------------- |
| Data Type           | Images      | Sequential             | Sequential             |
| Memory              | No          | Short-term             | Long-term              |
| Parallel Processing | Yes         | No (across time steps) | No (across time steps) |
| Long Dependency     | No          | Weak                   | Strong                 |
| Main Mechanism      | Convolution | Hidden State           | Gates + Cell State     |

### Self-Attention: Query, Key, Value

Self-attention is the core mechanism behind Transformers. Self-Attention allows each token to examine other tokens in the same sequence and decide which ones are most relevant.

| Vector | Full Name | Purpose                          | Easy Analogy   |
| ------ | --------- | -------------------------------- | -------------- |
| **Q**  | Query     | What am I looking for?           | Search request |
| **K**  | Key       | What information do I represent? | Search label   |
| **V**  | Value     | What information do I provide?   | Actual content |

### Transformer Main Components

| Component            | Meaning                     |
| -------------------- | --------------------------- |
| Tokenization         | Convert text to tokens      |
| Embedding            | Convert tokens to vectors   |
| Positional Encoding  | Add order information       |
| Self-Attention       | Tokens look at other tokens |
| Multi-Head Attention | Multiple attention views    |
| Feed-Forward Network | Further transformation      |
| Layer Normalization  | Stabilizes training         |
| Residual Connection  | Helps gradient flow         |
| Encoder              | Understands input           |
| Decoder              | Generates output            |



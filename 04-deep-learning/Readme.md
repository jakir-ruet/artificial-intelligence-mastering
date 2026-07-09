### Deep Learning

Deep Learning is a subset of Machine Learning that uses multi-layer artificial neural networks to learn hierarchical representations and complex patterns from data, typically through forward propagation, loss computation, back propagation, and optimization. Deep Learning is a subset of Machine Learning that uses multi-layer neural networks.

**Key relationship**

```bash
Artificial Intelligence
        ↓
Machine Learning
        ↓
Deep Learning
        ↓
Neural Networks
```

| SL  | Topic                   | Key Concept                            | Example                        |
| --- | ----------------------- | -------------------------------------- | ------------------------------ |
| 1   | Deep Learning Basics    | Neural networks learn complex patterns | Student performance prediction |
| 2   | Neuron/Perceptron       | Input → weight → bias → output         | Risk score calculation         |
| 3   | Activation Functions    | ReLU, Sigmoid, Softmax                 | Convert score to prediction    |
| 4   | Forward Propagation     | Data moves through network             | Input → hidden layers → output |
| 5   | Loss Function           | Measures prediction error              | Wrong dropout prediction       |
| 6   | Backpropagation         | Adjusts weights using error            | Model learns from mistakes     |
| 7   | Gradient Descent        | Optimization algorithm                 | Reduce loss step by step       |
| 8   | Neural Network Training | Epoch, batch, learning rate            | Train model repeatedly         |
| 9   | CNN                     | Image-based learning                   | Exam paper image analysis      |
| 10  | RNN/LSTM                | Sequence learning                      | Attendance trend prediction    |
| 11  | PyTorch Basics          | Practical DL framework                 | Build neural network           |
| 12  | End-to-End Project      | Complete DL pipeline                   | Student performance predictor  |

> - Convolutional Neural Network (CNN)
> - Recurrent Neural Network (RNN)
> - Long Short-Term Memory (LSTM)

Because the neural network contains multiple computational layers.

### Deep Learning Basics

Deep Learning is a subset of Machine Learning that uses multi-layer artificial neural networks to learn complex patterns and hierarchical representations from data.

| Topic                   | Meaning                                | Example                      |
| ----------------------- | -------------------------------------- | ---------------------------- |
| **Neural Network**      | Connected layers of artificial neurons | Student risk model           |
| **Input Layer**         | Receives features                      | Attendance, marks            |
| **Hidden Layer**        | Learns patterns                        | Academic-risk relationships  |
| **Output Layer**        | Produces prediction                    | Dropout probability          |
| **Neuron**              | Basic computational unit               | Weighted feature calculation |
| **Weight**              | Learned importance of input            | Attendance importance        |
| **Bias**                | Adjustable offset                      | Shifts decision boundary     |
| **Activation Function** | Adds non-linearity                     | ReLU, Sigmoid                |
| **Forward Propagation** | Input → prediction                     | Student data → risk          |
| **Loss Function**       | Measures error                         | Actual vs predicted          |
| **Back Propagation**    | Computes gradients                     | Learn from prediction error  |
| **Optimizer**           | Updates parameters                     | SGD, Adam                    |
| **Epoch**               | One full pass over training data       | Process all students once    |
| **Batch**               | Small subset of training data          | 32 students at a time        |
| **Learning Rate**       | Parameter update step size             | `0.001`                      |

#### Basic Neural Network Architecture (Why is it called “Deep”?)

```bash
Input Layer
    ↓
Hidden Layer 1
    ↓
Hidden Layer 2
    ↓
Output Layer
```

#### How a Neuron Works

A neuron is the basic computational unit of a neural network in Deep Learning.

**Core Formula**

```bash
z = (x₁w₁ + x₂w₂ + ... + xₙwₙ) + b
output = activation(z)
```

Here

| Symbol       | Meaning               | Example            |
| ------------ | --------------------- | ------------------ |
| `x`          | Input / feature       | Attendance         |
| `w`          | Weight                | Feature importance |
| `b`          | Bias                  | Adjustable offset  |
| `z`          | Weighted sum          | Raw neuron score   |
| `activation` | Output transformation | Sigmoid            |

#### How Deep Learning Learns

```bash
1. Input Data
      ↓
2. Forward Propagation
      ↓
3. Prediction
      ↓
4. Calculate Loss
      ↓
5. Backpropagation
      ↓
6. Compute Gradients
      ↓
7. Optimizer Updates Weights
      ↓
8. Repeat
```

#### Common Activation Functions

| Function         | Formula              | Output Range            | Common Use                       | Main Strength                                   | Main Limitation                                       | Example                   |
| ---------------- | -------------------- | ----------------------- | -------------------------------- | ----------------------------------------------- | ----------------------------------------------------- | ------------------------- |
| **ReLU**         | `max(0, x)`          | `0 → ∞`                 | Hidden layers                    | Fast, simple, reduces vanishing-gradient issues | Dying ReLU                                            | CNN / MLP hidden layers   |
| **Sigmoid**      | `1 / (1 + e⁻ˣ)`      | `0 → 1`                 | Binary classification output     | Probability-like output                         | Vanishing gradients, not zero-centered                | Dropout: Yes/No           |
| **Tanh**         | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)`  | `-1 → 1`                | Some sequence/recurrent networks | Zero-centered                                   | Vanishing gradients                                   | RNN hidden states         |
| **Softmax**      | `eˣⁱ / Σeˣʲ`         | Each `0 → 1`; sum = `1` | Multi-class output               | Produces class distribution                     | Sensitive to large logits; not for independent labels | Low/Medium/High           |
| **Leaky ReLU**   | `x if x>0 else αx`   | `-∞ → ∞`                | Hidden layers                    | Reduces dying ReLU                              | Extra slope choice `α`                                | Deep networks             |
| **GELU**         | Smoothly gates input | `≈ -0.17 → ∞`           | Transformers                     | Smooth, strong empirical performance            | More compute than ReLU                                | BERT-style models         |
| **SiLU / Swish** | `x · sigmoid(x)`     | `≈ -0.28 → ∞`           | Modern deep networks             | Smooth gradients                                | More compute than ReLU                                | Vision/deep architectures |

#### Major Deep Learning Architectures

| Architecture    | Full Name                                          | Main Use                     | Example                |
| --------------- | -------------------------------------------------- | ---------------------------- | ---------------------- |
| **ANN / MLP**   | Artificial Neural Network / Multi-Layer Perceptron | Tabular data                 | Dropout prediction     |
| **CNN**         | Convolutional Neural Network                       | Images                       | Exam paper recognition |
| **RNN**         | Recurrent Neural Network                           | Sequential data              | Attendance sequence    |
| **LSTM**        | Long Short-Term Memory                             | Long sequences               | Performance trends     |
| **Transformer** | Transformer Neural Network                         | Language and multimodal data | LLMs                   |

### Machine Learning vs Deep Learning

| Machine Learning                          | Deep Learning                           |
| ----------------------------------------- | --------------------------------------- |
| Often requires manual feature engineering | Can learn representations automatically |
| Effective on smaller structured datasets  | Often benefits from large datasets      |
| Lower compute requirements                | Higher compute requirements             |
| Decision Tree, Random Forest              | CNN, RNN, Transformer                   |
| Strong for tabular data                   | Strong for image, text, audio, video    |
| Works with smaller data                   | Usually needs more data                 |
| Faster to train                           | More compute-heavy                      |
| Example: Logistic Regression              | Example: Neural Network                 |

> Deep Learning = Machine Learning using deep neural networks

### Neuron/Perceptron

A neuron is the basic computational unit of a neural network in Deep Learning. An artificial neuron receives input features, multiplies them by learned weights, adds a bias term, and passes the resulting weighted sum through an activation function to produce an output.

> The Perceptron is a simple linear binary classifier.

#### Biological Neuron vs Artificial Neuron

- Biological Neuron

```bash
Biological Neuron

Dendrites
   ↓
Cell Body
   ↓
Axon
   ↓
Output Signal
```

- Artificial Neuron

```bash
Input Features
   ↓
Weights
   ↓
Weighted Sum
   ↓
Bias
   ↓
Activation Function
   ↓
Output
```

| Biological       | Artificial   |
| ---------------- | ------------ |
| Dendrites        | Inputs       |
| Synapses         | Weights      |
| Cell body        | Weighted sum |
| Firing mechanism | Activation   |
| Axon             | Output       |

**The neuron computes**

```bash
Input x₁ ──× w₁──┐
                 │
Input x₂ ──× w₂──┼──> Sum + Bias ──> Activation ──> Output
                 │
Input x₃ ──× w₃──┘
```

```bash
z = x₁w₁ + x₂w₂ + x₃w₃ + b
output = f(z)
```

| Symbol   | Meaning                       |
| -------- | ----------------------------- |
| `xᵢ`     | Input feature                 |
| `wᵢ`     | Weight                        |
| `b`      | Bias                          |
| `z`      | Weighted sum / pre-activation |
| `f`      | Activation function           |
| `output` | Neuron activation             |

> - Bias is an additional trainable parameter
> - A weight represents the learned influence of an input.

#### Neuron vs Perceptron vs MLP

| Concept               | Meaning                    | Activation          | Capability                  |
| --------------------- | -------------------------- | ------------------- | --------------------------- |
| **Artificial Neuron** | General computational unit | ReLU, Sigmoid, etc. | Building block              |
| **Perceptron**        | Linear binary classifier   | Step function       | Linearly separable problems |
| **MLP**               | Multi-Layer Perceptron     | ReLU/Sigmoid/etc.   | Nonlinear complex patterns  |

### Activation Function

An activation function determines how a neuron transforms its raw weighted sum. For binary classification, we can use the Sigmoid function.

```bash
σ(z) = 1 / (1 + e⁻ᶻ)
```

> Key Takeaway

```bash
Neuron = Inputs + Weights + Bias + Activation
```

### Forward Propagation

Forward propagation is the process of passing input data forward through a neural network, layer by layer, to produce a prediction.

```bash
Input → Weighted Sum → Activation → Hidden Layers → Output → Prediction
```

### Loss Function

A Loss function measures how wrong a model’s prediction is compared with the actual target.

> Loss Function = numerical measure of prediction error

#### Common Loss Functions

| Problem Type               | Loss Function        | Typical Use                   |
| -------------------------- | -------------------- | ----------------------------- |
| Binary Classification      | Binary Cross-Entropy | Dropout Yes/No                |
| Multi-Class Classification | Cross-Entropy        | Low/Medium/High               |
| Regression                 | Mean Squared Error   | Predict marks                 |
| Regression                 | Mean Absolute Error  | Predict salary/score robustly |
| Imbalanced Classification  | Focal Loss           | Rare-event detection          |

> - **Mean Squared Error** `MSE = (1/n) Σ(yᵢ - ŷᵢ)²`
> - **Mean Absolute Error** `MAE = (1/n) Σ|yᵢ - ŷᵢ|`

| Feature             | MSE                 | MAE                                |
| ------------------- | ------------------- | ---------------------------------- |
| Error treatment     | Squares errors      | Absolute errors                    |
| Large errors        | Strong penalty      | Linear penalty                     |
| Outlier sensitivity | High                | Lower                              |
| Smooth derivative   | Yes                 | Not differentiable exactly at zero |
| Common use          | Standard regression | Robust regression                  |

**Loss Function and Output Activation**

| Problem                    | Output Layer         | Typical Loss  |
| -------------------------- | -------------------- | ------------- |
| Binary Classification      | Sigmoid              | BCE           |
| Multi-Class Classification | Softmax conceptually | Cross-Entropy |
| Multi-Label Classification | Independent Sigmoids | BCE           |
| Regression                 | Linear output        | MSE / MAE     |

### Back Propagation

Backpropagation is the algorithm used to compute how the loss changes with respect to every trainable weight and bias in a neural network.

### Gradient Descent

A gradient tells how much and in which local direction the loss changes when a model parameter, such as a weight or bias, changes. Formula - For one weight:

```bash
∂L
──
∂w
```

Where,

- L = Loss
- w = Weight
- ∂L/∂w = Gradient of loss with respect to weight

| Concept             | Analogy                | Technical Meaning                      |
| ------------------- | ---------------------- | -------------------------------------- |
| Forward propagation | Student answers        | Model predicts                         |
| Loss                | Teacher marks mistake  | Error value                            |
| Gradient            | Correction feedback    | Direction and size of parameter change |
| Backpropagation     | Feedback sent backward | Compute gradients layer by layer       |
| Optimizer           | Student adjusts method | Updates weights                        |

**One-line memory**
- **Gradient** = feedback
- **Backpropagation** = process of calculating and sending that feedback backward
- **Optimizer** = uses feedback to update weights

### Neural Network Training

Neural Network Training is the iterative process of adjusting a network’s weights and biases so that its predictions become more accurate and its loss decreases. Core Training Terms

| Term                 | Definition                                        | Example                           |
| -------------------- | ------------------------------------------------- | --------------------------------- |
| **Epoch**            | One complete pass through the training dataset    | All 1,000 students processed once |
| **Batch**            | Subset of training samples                        | 32 students                       |
| **Batch Size**       | Number of samples in one batch                    | `32`                              |
| **Iteration / Step** | One parameter update                              | One batch → one update            |
| **Learning Rate**    | Controls update step size                         | `0.001`                           |
| **Loss**             | Measures prediction error                         | BCE loss                          |
| **Gradient**         | Local sensitivity of loss to parameters           | `∂L/∂w`                           |
| **Backpropagation**  | Computes gradients                                | `loss.backward()`                 |
| **Optimizer**        | Uses gradients to update parameters               | Adam                              |
| **Validation**       | Evaluates during training on unseen held-out data | Validation F1                     |

### Convolutional Neural Network (CNN)

A Convolutional neural network, or CNN, is a deep-learning architecture designed to automatically learn spatial patterns from grid-like data, especially images.

> CNN = Neural Network specialized for detecting local patterns such as edges, textures, shapes, and objects.

#### Common CNN Tasks

| Task                 | Example                         |
| -------------------- | ------------------------------- |
| Image Classification | Cat vs Dog                      |
| Object Detection     | Find students in classroom      |
| Image Segmentation   | Separate objects pixel-by-pixel |
| OCR                  | Read exam papers                |
| Face Recognition     | Identify person                 |
| Medical Imaging      | Detect abnormalities            |
| Document AI          | Analyze forms and invoices      |

### Recurrent Neural Network (RNN)/Long Short-Term Memory (LSTM)

- A Recurrent neural network (RNN) is a neural-network architecture designed for sequential data, where the current computation depends on both the current input and information carried from previous time steps.
- A Long short-term memory (LSTM) is a specialized type of RNN designed to preserve useful information over longer sequences and mitigate the vanishing-gradient problem.

> - RNN = sequence processing with recurrent hidden state
> - LSTM = RNN with gated memory control

**LSTM Main Components**

| Component           | Purpose                          |
| ------------------- | -------------------------------- |
| **Forget Gate**     | What old information to remove   |
| **Input Gate**      | What new information to write    |
| **Candidate State** | Proposed new memory content      |
| **Cell State**      | Long-term memory pathway         |
| **Output Gate**     | What memory to expose            |
| **Hidden State**    | Current recurrent representation |

**RNN vs LSTM**

| Feature                       | RNN     | LSTM     |
| ----------------------------- | ------- | -------- |
| Sequential data               | Yes     | Yes      |
| Hidden state                  | Yes     | Yes      |
| Cell state                    | No      | Yes      |
| Gates                         | No      | Yes      |
| Long-range dependencies       | Weaker  | Better   |
| Vanishing-gradient resistance | Limited | Improved |
| Complexity                    | Lower   | Higher   |
| Parameters                    | Fewer   | More     |

**RNN vs LSTM vs CNN**

| Model       | Best Suited For                       | Example                    |
| ----------- | ------------------------------------- | -------------------------- |
| CNN         | Spatial patterns                      | Student ID image           |
| RNN         | Shorter/simple sequences              | Attendance sequence        |
| LSTM        | Longer dependencies                   | Multi-semester performance |
| Transformer | Long-range sequence modeling at scale | LLMs                       |

#### Complete Deep Learning Connection

```bash
Input Features
      ↓
Artificial Neurons
      ↓
Layer
      ↓
Multiple Layers
      ↓
Deep Neural Network
      ↓
Forward Propagation
      ↓
Prediction
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Optimizer
      ↓
Updated Weights and Biases
```

### PyTorch Basics





## Large Language Model - `LLM`

A Large language model (LLM) is a large neural network, typically based on the Transformer architecture, trained on massive datasets to model language and perform tasks such as generation, summarization, translation, question answering, coding, and reasoning-oriented workflows.

> LLM = Transformer + Large-Scale Data + Large Parameter Count + Training + Adaptation

| Model Family | Creator        | Type              | Main Strength                           | Local Running | Customization | Privacy   | Multimodal |
| ------------ | -------------- | ----------------- | --------------------------------------- | ------------- | ------------- | --------- | ---------- |
| GPT          | OpenAI         | Closed-source     | General intelligence, reasoning, coding | No            | Medium        | Medium    | Excellent  |
| LLaMA        | Meta Platforms | Open-weight       | Local AI, customization, research       | Yes           | Excellent     | Excellent | Improving  |
| Claude       | Anthropic      | Closed-source     | Writing, analysis, long documents       | No            | Low           | Medium    | Good       |
| Gemini       | Google         | Closed-source     | Multimodal AI, Google ecosystem         | No            | Low           | Medium    | Excellent  |
| Mistral      | Mistral AI     | Open & commercial | Efficiency, deployment flexibility      | Yes           | Good          | Good      | Limited    |

- **LLaMA** > Large Language Model Meta AI
- **GPT** > Generative Pre-trained Transformer

> Key Characteristics
>
> - `Large` → trained on billions or trillions of words
> - `Language` → works with human text (English, Bangla, etc.)
> - `Model` → a mathematical neural network that learns patterns

### Chef Analogy - Restaurant

Imagine a famous chef. The chef has cooked:

- Pizza
- Pasta
- Curry
- Sushi
- Burgers
- Desserts

**Now someone says:** *Make me something spicy with chicken.* **The chef doesn't search a recipe book every time.** Instead, the chef combines experience from thousands of previous dishes.

> An LLM works similarly. It doesn't retrieve a fixed answer from memory. It generates a response based on patterns learned during training.

### What an LLM cannot reliably do?

An LLM is **not** a perfect source of truth. It can:

- Make factual mistakes/**Hallucinations**
- Be unaware of events after its training unless connected to external tools
- Produce confident-sounding but incorrect answers
- Misunderstand ambiguous prompts

> For tasks that require current or authoritative information, LLMs are often combined with search engines or databases.

### LLM vs Search Engine

| LLM                              | Search Engine                   |
| -------------------------------- | ------------------------------- |
| Generates an answer              | Finds relevant documents        |
| Learns from training data        | Searches indexed web pages      |
| Can explain and reason over text | Primarily retrieves information |
| May make mistakes                | Can point to original sources   |

> Many modern AI assistants combine both: the LLM generates the response while a search system retrieves up-to-date information when needed.

### How an LLM (e.g., GPT) Generates a Response

| Step                                               | What Happens                                                                                   | Example                                                                       |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **1. User Prompt**                                 | User sends input text                                                                          | `"Explain Docker in simple terms."`                                           |
| **2. Tokenization**                                | Text is split into tokens and converted into numbers (IDs)                                     | `"Explain Docker"` → `[1245, 9821, 304]`                                      |
| **3. Embedding + Transformer Processing**          | Tokens are converted into vectors and processed through attention layers to understand context | Model understands that *“Docker” = technology*, *“simple” = easy explanation* |
| **4. Context Understanding (Attention Mechanism)** | Model focuses on important words and relationships                                             | “simple” affects explanation style                                            |
| **5. Probability Prediction**                      | Model calculates probability of next possible token                                            | `P("is") = 0.35`, `P("Docker") = 0.20`, `P("a") = 0.25`                       |
| **6. Next Token Selection**                        | One token is selected (highest probability or sampling)                                        | `"Docker"`                                                                    |
| **7. Iterative Generation**                        | Steps 4–6 repeat to build full sentence                                                        | `"Docker is a platform that..."`                                              |
| **8. Final Response Output**                       | Complete response is returned to user                                                          | Full explanation appears                                                      |

> - `User Prompt → Tokenization → Embedding → Transformer (Attention) → Probability Prediction → Token Generation → Full Response`

### LLaMA + Ollama + Open WebUI - Install - Recommended

```bash
ollama pull llama3.2
ollama list
ollama run llama3.2
```

```bash
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

```bash
http://localhost:3000


### LLM Learning Flow

| Step | Topic                    | Example                          |
| ---- | ------------------------ | -------------------------------- |
| 1    | LLM Fundamentals         | What is an LLM?                  |
| 2    | Tokenization             | `"AI is great"` → tokens         |
| 3    | Embeddings               | Token → vector                   |
| 4    | Transformer Architecture | Attention blocks                 |
| 5    | Next-Token Prediction    | `"AI is"` → `"powerful"`         |
| 6    | Pretraining              | Learn broad language patterns    |
| 7    | Context Window           | Prompt + history                 |
| 8    | Inference                | Generate tokens                  |
| 9    | Decoding                 | Temperature, Top-k, Top-p        |
| 10   | Prompt Engineering       | Zero-shot, few-shot              |
| 11   | Fine-Tuning              | Domain adaptation                |
| 12   | Instruction Tuning       | Follow instructions              |
| 13   | Preference Alignment     | Human/AI preference optimization |
| 14   | Quantization             | FP16, INT8, INT4                 |
| 15   | LLM Serving              | API inference                    |
| 16   | Evaluation               | Quality, safety, latency         |
| 17   | Production Architecture  | Cloud deployment                 |

#### Training vs Inference

| Training                  | Inference                   |
| ------------------------- | --------------------------- |
| Learn parameters          | Use learned parameters      |
| Huge datasets             | User prompt                 |
| Forward + backward        | Usually forward only        |
| Compute gradients         | No gradient updates         |
| Optimizer updates weights | Generate output             |
| GPU/TPU clusters          | GPU/CPU/accelerator serving |

#### LLM Lifecycle

```bash
Raw Data
   ↓
Cleaning & Filtering
   ↓
Tokenization
   ↓
Pretraining
   ↓
Base Model
   ↓
Instruction Tuning
   ↓
Preference / Alignment Training
   ↓
Evaluation
   ↓
Optimization
   ↓
Deployment
   ↓
Monitoring
```

### Main Tokenization Types

| Type            | Example                | Main Issue / Strength    |
| --------------- | ---------------------- | ------------------------ |
| Word-level      | `["I","love","AI"]`    | Huge vocabulary          |
| Character-level | `["A","I"]`            | Long sequences           |
| Subword         | `["learn","ing"]`      | Strong practical balance |
| Byte-level      | Raw byte-derived units | Robust coverage          |

### Common Tokenization Algorithms

| Algorithm     | Full Name / Idea                       |
| ------------- | -------------------------------------- |
| BPE           | Byte Pair Encoding                     |
| WordPiece     | Subword vocabulary approach            |
| Unigram       | Probabilistic subword segmentation     |
| SentencePiece | Language-independent tokenizer toolkit |

### Embeddings

An embedding is a trainable dense vector representation of a token. It maps discrete token IDs into continuous vector space, allowing a neural network to capture semantic relationships and perform mathematical operations on language.

```bash
Word
↓
Token ID
↓
Embedding
↓
Vector
↓
Transformer
```

Simply

```bash
Embedding
=
Meaningful Vector Representation
```

**Tokenization**

| Token |   ID |
| ----- | ---: |
| I     |    0 |
| love  |    1 |
| AI    |    2 |

**Embedding matrix**

| Token | Vector (illustrative) |
| ----- | --------------------- |
| I     | [0.12, -0.44, 0.88]   |
| love  | [-0.51, 0.90, 0.16]   |
| AI    | [0.73, -0.11, 0.64]   |

> - The model learns these vectors automatically.

### Next-Token Prediction

Next-token prediction is the core training and inference objective of autoregressive language models. Given all previous tokens, the model estimates a probability distribution over the vocabulary and predicts the most likely next token, repeating this process to generate complete sequences.

Mathematically `P(next token | previous tokens)`

**Example**

Sentence `I love artificial`

The model predicts

| Candidate    | Probability |
| ------------ | ----------: |
| intelligence |        0.82 |
| flowers      |        0.08 |
| pizza        |        0.03 |
| bananas      |        0.01 |
| ...          |         ... |

### The Entire LLM Training Pipeline

```bash
Raw Internet Data
        ↓
Cleaning
        ↓
Filtering
        ↓
Tokenization
        ↓
Pretraining
        ↓
Base Model
        ↓
Instruction Tuning
        ↓
Alignment
        ↓
Production LLM
```

### LLM Inference

Inference is the process of using a trained language model to generate predictions. During inference, the model performs only forward computation, producing one token at a time without updating its parameters.

> Simply:
> - Training = Learn
> - Inference = Use

| Training                | Inference                |
| ----------------------- | ------------------------ |
| Learn from data         | Generate answers         |
| Updates model weights   | Weights remain unchanged |
| Forward + Backward pass | Forward pass only        |
| Computes gradients      | No gradients             |
| Uses optimizer          | No optimizer             |
| Uses loss function      | No loss function         |

**During Training**

```bash
Input
   ↓
Transformer
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Update Weights
```

**During Inference**

```bash
Prompt
   ↓
Tokenizer
   ↓
Transformer
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
```

### Sampling

The process of selecting the next token from the probability distribution produced by an LLM.

| Aspect             | Description                                           |
| ------------------ | ----------------------------------------------------- |
| **Purpose**        | Controls response quality, diversity, and creativity. |
| **Common Methods** | Greedy, Temperature, Top-k, Top-p (Nucleus).          |
| **Output**         | Determines how the LLM generates text.                |

```bash
User Prompt
      ↓
Tokenizer
      ↓
Transformer
      ↓
Logits
      ↓
Softmax
      ↓
Sampling
      ↓
Next Token
      ↓
Repeat
```

### Prompt Engineering

It's the practice of designing effective instructions, examples, constraints, and output formats that guide an LLM toward the desired behavior without changing the model's parameters.

Notice that prompt engineering isn't just the **recipe**; it also includes:

- Task instructions
- Role assignment
- Constraints
- Examples (few-shot)
- Output schemas (JSON, Markdown, XML)
- Tool usage instructions (in AI applications)

#### Prompt Types

| Category           | Example                                       |
| ------------------ | --------------------------------------------- |
| Instruction Prompt | Explain Docker.                               |
| Question Prompt    | What is Kubernetes?                           |
| Zero-shot          | Translate "Hello" to Spanish.                 |
| One-shot           | Cat → Animal; Dog → ?                         |
| Few-shot           | Apple → Fruit, Carrot → Vegetable, Banana → ? |
| Role Prompt        | You are a DevOps engineer.                    |
| Structured Prompt  | Return JSON only.                             |
| Chain-of-Thought*  | Solve step by step.                           |

### Context Engineering

It's the process of selecting, organizing, and delivering the most relevant information to the model for a specific request.

Because context engineering includes actively deciding:

- Which documents to retrieve
- Which chat history to include
- Which memories to inject
- Which examples to remove
- Which tools to call
- How to fit everything into the context window

| Prompt                                  | Context                   |
| --------------------------------------- | ------------------------- |
| What you ask                            | Everything the model sees |
| Active instruction                      | Available information     |
| Usually written by the user/application | Built by the application  |
| Small                                   | Can be much larger        |

### Learning Roadmap

| Stage           | Prompt     | Context    |
| --------------- | ---------- | ---------- |
| AI Fundamentals | No         | No         |
| ML              | No         | No         |
| Deep Learning   | No         | No         |
| Transformers    | No         | No         |
| LLMs            | Yes        | No         |
| Generative AI   | Yes        | Partial    |
| Embeddings      | Yes        | Partial    |
| RAG             | Yes        | Yes        |
| Agentic AI      | Advanced   | Advanced   |
| AI Engineering  | Advanced   | Advanced   |
| LLMOps          | Production | Production |
| Cloud AI        | Enterprise | Enterprise |

### Fine-Tuning

The process of continuing the training of a pretrained model on domain- or task-specific data to improve its performance for a particular use case.

| Field              | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| **Purpose**        | Specializes the model for a specific domain, task, or behavior. |
| **Common Methods** | Full Fine-Tuning, PEFT, LoRA, QLoRA.                            |
| **Output**         | A specialized version of the pretrained model.                  |

```bash
Pretrained Model
        ↓
Task-Specific Data
        ↓
Forward Pass
        ↓
Loss
        ↓
Backpropagation
        ↓
Update Weights
        ↓
Specialized Model
```

### Alignment

The process of training an LLM to better align its responses with human preferences, instructions, and safety objectives.

| Field              | Description                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**        | Makes the model more helpful, truthful, and safe.                                                                         |
| **Common Methods** | Supervised Fine-Tuning (SFT), RLHF, DPO, Safety Alignment.                                                                |
| **Output**         | An instruction-following and safety-aligned AI assistant.                                                                 |

```bash
Pretrained Model
        ↓
Supervised Fine-Tuning
        ↓
Preference Optimization
        ↓
Safety Alignment
        ↓
Aligned AI Assistant
```

### Quantization

| Field                        | Description                                                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**                  | Reduces memory usage, speeds up inference, and lowers deployment costs.                                                               |
| **Common Methods / Formats** | FP16, BF16, INT8, INT4, GPTQ, AWQ, GGUF.                                                                                              |
| **Output**                   | A smaller, faster, and more efficient model for deployment.                                                                           |

```bash
Trained Model
      ↓
Quantization
      ↓
INT8 / INT4
      ↓
Smaller Model
      ↓
Faster Inference
      ↓
Deployment
```

### Distillation

Knowledge distillation is a model compression technique in which a smaller student model is trained to imitate the outputs or behavior of a larger teacher model, producing a faster and more efficient model while retaining much of the teacher's capability. Knowledge distillation is the process of training a smaller model (student) to imitate the behavior of a larger, more capable model (teacher).

> Simply: Big Model → Teaches → Small Model

```bash
Question
↓
Teacher Model
↓
High-Quality Answer
↓
Student Learns
```

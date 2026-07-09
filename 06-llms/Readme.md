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

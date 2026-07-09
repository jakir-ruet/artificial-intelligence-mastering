### Transformer

A Transformer is a neural-network architecture based on self-attention, allowing tokens in a sequence to attend to each other in parallel and learn contextual relationships. It is the foundation of modern large language models.

> Transformer = Attention-based neural network architecture for understanding relationships between tokens.

#### Why Transformers? Before Transformers:

| Model | Problem                                               |
| ----- | ----------------------------------------------------- |
| RNN   | Slow for long sequences                               |
| LSTM  | Better memory, but still sequential                   |
| CNN   | Good local patterns, weaker global sequence reasoning |

> Transformers solve this by looking at many tokens in parallel.

#### Transformer Main Components

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

#### Why Transformers are powerful

| Strength            | Explanation                                     |
| ------------------- | ----------------------------------------------- |
| Parallel processing | Faster than RNN sequence-by-sequence processing |
| Long-range context  | Can connect distant tokens                      |
| Scalable            | Works well with large datasets and models       |
| Foundation for LLMs | GPT, BERT, T5, Llama-style models               |

### Self-Attention: Query, Key, Value

Self-attention is the core mechanism behind Transformers. Self-Attention allows each token to examine other tokens in the same sequence and decide which ones are most relevant.

| Vector | Full Name | Purpose                          | Easy Analogy   |
| ------ | --------- | -------------------------------- | -------------- |
| **Q**  | Query     | What am I looking for?           | Search request |
| **K**  | Key       | What information do I represent? | Search label   |
| **V**  | Value     | What information do I provide?   | Actual content |

### Embedding

An embedding is a numerical vector that represents the meaning (semantic information) of data.

The data can be:

- Text
- Images
- Audio
- Video
- Code

> Simply: `Embedding = Meaning converted into numbers.`

For example:

```bash
I love Java.
73 20 6C 6F 76 65 ... # The computer sees
[0.18, -0.91, 0.47, ..., 0.65] # An embedding model converts it into something like
```

| Model                 | Dimensions |
| --------------------- | ---------: |
| Small embedding model |        384 |
| Medium                |        768 |
| Large                 |       1024 |
| Larger                | 1536–3072+ |

### LLM vs Embedding Model

| LLM                 | Embedding Model       |
| ------------------- | --------------------- |
| Generates text      | Generates vectors     |
| Predicts next token | Encodes meaning       |
| Chat                | Search                |
| Writes answers      | Finds similar content |

> Think of them as complementary:
> - Embedding model: Finds relevant information.
> - LLM: Uses that information to generate a response.

### Where Embeddings Are Used

| Application            | Why Embeddings?           |
| ---------------------- | ------------------------- |
| Semantic Search        | Find documents by meaning |
| RAG                    | Retrieve relevant context |
| Recommendation Systems | Match similar items       |
| Chatbots               | Retrieve knowledge        |
| AI Agents              | Find memories or tools    |
| Duplicate Detection    | Identify similar content  |
| Document Clustering    | Group related documents   |

### Semantic Search

Semantic search is a search technique that retrieves information based on meaning rather than exact keyword matching.

> Simply: `Semantic Search = Search by Meaning, Not by Words`

**Keyword Search vs Semantic Search**

| Keyword Search             | Semantic Search      |
| -------------------------- | -------------------- |
| Exact words                | Meaning              |
| Misses synonyms            | Understands synonyms |
| Literal matching           | Conceptual matching  |
| Simple index               | Vector database      |
| Traditional search engines | Modern AI search     |

### Vector Database

A vector database is a database designed to `store`, `index`, and search embedding vectors efficiently.

- Unlike a traditional relational database, it doesn't primarily answer: `Find rows where student_id = 1001.`
- Instead, it answers: `Find the vectors most similar to this vector.`

| Traditional Database       | Vector Database                 |
| -------------------------- | ------------------------------- |
| Stores rows and columns    | Stores high-dimensional vectors |
| SQL queries                | Similarity search               |
| Exact matches              | Semantic matches                |
| Equality, joins, filters   | Nearest-neighbor search         |
| Optimized for transactions | Optimized for vector retrieval  |

### What Does a Vector Database Do?

A vector database stores vectors and builds specialized indexes so it can quickly find the nearest vectors.

Conceptually:

```bash
Documents
      │
      ▼
Embedding Model
      │
      ▼
Vectors
      │
      ▼
Vector Database
      │
      ▼
Fast Similarity Search
```

**What Is Stored?**

| Document                   | Vector               | Metadata              |
| -------------------------- | -------------------- | --------------------- |
| Student Registration Guide | `[0.21, -0.45, ...]` | Department=Admissions |
| Attendance Policy          | `[0.37, 0.14, ...]`  | Department=Academic   |
| Payroll Rules              | `[-0.63, 0.58, ...]` | Department=HR         |

**Query Flow**

1. Suppose a user asks

```bash
How do I register a student?
```

2. Pipeline

```bash
Question
      │
      ▼
Embedding Model
      │
      ▼
Query Vector
      │
      ▼
Vector Database
      │
      ▼
Nearest Documents
      │
      ▼
LLM
      │
      ▼
Answer
```

**SQL Database vs. Vector Database**

| SQL Database                        | Vector Database                 |
| ----------------------------------- | ------------------------------- |
| `SELECT * FROM students WHERE id=1` | Find the 5 most similar vectors |
| Exact values                        | Semantic similarity             |
| B-tree indexes                      | Vector indexes                  |
| Transactions                        | Similarity retrieval            |
| Business data                       | AI knowledge retrieval          |

> Both often work together in enterprise systems:
> - SQL database → transactional/business data
> - Vector database → semantic retrieval

### Exact vs Approximate

| Exact Nearest Neighbor (ENN)  | Approximate Nearest Neighbor (ANN) |
| ----------------------------- | ---------------------------------- |
| Finds the true nearest vector | Finds a very close vector          |
| Slower at scale               | Much faster                        |
| High computational cost       | Lower computational cost           |
| Useful for small datasets     | Ideal for production AI systems    |

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an AI architecture that combines semantic retrieval with a large language model. Relevant documents are retrieved from an external knowledge source and provided as context to the LLM, enabling more accurate, current, and organization-specific responses.

It combines two capabilities:

1. **Retrieval** → Find relevant information.
2. **Generation** → Use an LLM to generate an answer from that information.

> Unlike a standalone LLM, a RAG system does not rely only on what the model learned during training.

```bash
User Question
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
Top-K Relevant Chunks
       │
       ▼
Prompt Construction
       │
       ▼
LLM
       │
       ▼
Final Answer
```

### Traditional LLM vs RAG

| Traditional LLM            | RAG                                |
| -------------------------- | ---------------------------------- |
| Uses training knowledge    | Uses retrieved knowledge           |
| Can become outdated        | Can use current documents          |
| Cannot access private data | Can answer from internal documents |
| Higher hallucination risk  | More grounded responses            |

### Why RAG Is So Powerful

RAG allows organizations to:

- Keep answers up to date without retraining the LLM.
- Use private or proprietary documents securely.
- Reduce hallucinations by grounding responses in retrieved context.
- Build AI assistants for internal knowledge bases, policies, manuals, APIs, and documentation.

### Complete RAG Architecture

The Big Picture

A production RAG system has two major pipelines:

1. Offline Pipeline (Indexing / Ingestion) – prepares documents.
2. Online Pipeline (Retrieval & Generation) – answers user questions.

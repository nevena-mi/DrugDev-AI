Implement only the requested module. Do not modify unrelated files or introduce functionality outside the current implementation step unless explicitly instructed.

# AGENT.md

# AI Learning & Intelligence Assistant for Drug Development & Regulatory Science

## Project Goal

Build an AI-powered learning and regulatory intelligence assistant using Retrieval-Augmented Generation (RAG). The application provides three modes: Ask, Learn and Monitor, using only authoritative documents from EMA, FDA, ICH, WHO and related organizations. The project prioritizes correctness, modularity and extensibility over feature count.

---

# Overall Architecture

Official Documents (PDFs)
        ↓
Document Ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Pinecone Vector Database
        ↓
Retriever
        ↓
LangGraph
        ↓
Ask | Learn | Monitor
        ↓
Streamlit UI

---

# Repository Structure

```
project/
│
├── project_plan.md
├── curriculum.yaml
├── requirements.txt
├── AGENT.md
│
├── sources/
│
├── data/
│
├── prompts/
│
├── tests/
│
└── src/
    ├── config.py
    ├── openai_client.py
    ├── pinecone_client.py
    ├── embeddings.py
    ├── ingest.py
    ├── retrieve.py
    ├── graph.py
    ├── curriculum.py
    ├── chatbot.py
    ├── quiz.py
    ├── monitor.py
    ├── metadata.py
    ├── app.py
    └── utils.py
```

The existing repository structure must be preserved.

---

# Development Strategy

Develop incrementally.

Implement one module at a time.

Each module must

- compile
- be independently testable
- pass its tests
- be committed before starting the next module

Do not implement future modules before the current module is complete.

---

# Implementation Order

Implement modules strictly in this order.

1. Configuration
2. Document ingestion
3. Embedding generation
4. Pinecone indexing
5. Retrieval
6. Curriculum loading
7. LangGraph workflow
8. Ask mode
9. Learn mode
10. Monitor mode
11. Streamlit UI

Do not skip steps.

---

# Coding Conventions

Use Python 3.12.
Use type hints.
Write concise docstrings for public functions.
Use the standard logging module.
Never print debugging information in production code.
Follow PEP 8.
Keep functions small and focused.
Avoid duplicated code.
Use pathlib instead of string file paths.

---

# Configuration

Secrets must exist only inside `.env`.

Never hardcode

- API keys
- model names
- index names
- paths

Load configuration exclusively through `config.py`.

---

# RAG Rules

Always retrieve relevant document chunks before generating an answer.
The LLM must answer only from retrieved context.
Every answer must include source citations.
If relevant information cannot be retrieved, state that the answer cannot be generated from the available documents.
Do not answer from model knowledge when retrieval fails.

---

# Curriculum Rules

Load the curriculum only from `curriculum.yaml`.

Never hardcode

- learning modules
- prerequisites
- learning objectives
- document mappings

The curriculum defines the learning order.

The LLM personalizes navigation but never modifies the curriculum.

---

# Document Rules

Treat the `sources/` directory as read-only.
Documents must never be modified.
Generated artifacts belong inside `data/`.

Document metadata must include

- filename
- source organization
- document title
- topic
- chunk id

---

# LangGraph Rules

Each node performs one responsibility only.
Keep graph nodes independent.
Graph state should remain minimal.
Avoid business logic inside the UI.

---

# Streamlit Rules

The interface contains exactly three modes.

- Ask
- Learn
- Monitor

The UI must not contain application logic.
All processing occurs in backend modules.

---

# Testing

Every module requires a corresponding test.
Tests must run independently.
Verify both expected behaviour and failure cases.
Test after every implementation step.
Do not continue development until the current module passes testing.

---


# Scope

Follow `project_plan.md` as the functional specification.

If implementation details are unclear, choose the simplest modular solution that supports future extension without changing the architecture.
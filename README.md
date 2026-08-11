# DrugDev-AI

**DrugDev-AI** is an AI-powered educational and regulatory assistant for pharmaceutical drug development. It combines Retrieval-Augmented Generation (RAG), structured learning, semantic search, and AI-assisted question answering to provide grounded information from authoritative regulatory sources.

The project was developed as part of the Ironhack AI Engineering Bootcamp.

---

# Features

## Learn Mode

An interactive curriculum covering major areas of pharmaceutical development:

* Drug Development Fundamentals
* Regulatory Landscape
* Clinical Trials
* Marketing Authorization
* Pharmaceutical Quality Systems
* Pharmacovigilance
* Emerging Topics

Each module provides:

* AI-generated learning lessons grounded in regulatory documents
* Learning objectives
* Module-specific Q&A
* Automatically generated quizzes
* Progress tracking

---

## Ask Mode

Ask natural-language questions about pharmaceutical development and regulatory science.

Answers are generated using Retrieval-Augmented Generation (RAG) and are grounded in authoritative regulatory documents rather than model memory alone.

Each answer includes document citations to improve transparency and traceability.

---

## Monitor Mode *(in progress)*

Planned functionality includes monitoring regulatory developments using external APIs, including:

* openFDA
* ClinicalTrials.gov
* EMA news and updates

---

# Architecture

```text
User
   │
   ▼
Question
   │
   ▼
OpenAI Embedding
   │
   ▼
Pinecone Vector Database
   │
   ▼
Top 15 Retrieved Chunks
   │
   ▼
Cohere Reranker
   │
   ▼
Top 5 Relevant Chunks
   │
   ▼
OpenAI Responses API
   │
   ▼
Grounded Answer
```

---

# Knowledge Base

The system currently indexes **37 regulatory and scientific documents** from major international organizations, including:

* ICH
* EMA
* FDA
* WHO
* European Union
* WMA
* GVP

Topics include:

* Good Clinical Practice
* Clinical Trials
* Pharmacovigilance
* Pharmaceutical Quality Systems
* Marketing Authorization
* Quality Risk Management
* Artificial Intelligence in Healthcare
* EU Pharmaceutical Regulation

The corpus contains approximately **6,645 vectorized text chunks** stored in Pinecone.

---

# Technology Stack

* Python
* Streamlit
* OpenAI Responses API
* OpenAI Embeddings
* Pinecone
* Cohere Rerank
* PyPDF
* YAML
* pytest

---

# Retrieval Pipeline

The application uses a two-stage retrieval architecture:

1. Semantic retrieval using Pinecone.
2. Neural reranking using Cohere.
3. Grounded response generation using OpenAI.

This improves ranking precision while preserving semantic recall.

---

# Retrieval Evaluation

Retrieval quality was evaluated using a fixed benchmark of representative regulatory questions before and after introducing Cohere reranking.

| Metric                     | Baseline | Reranked |
| -------------------------- | -------: | -------: |
| Hit@1                      |     0.40 | **0.80** |
| Hit@3                      |     0.80 |     0.80 |
| Hit@5                      |     0.80 |     0.80 |
| Mean Reciprocal Rank (MRR) |     0.60 | **0.80** |

Key findings:

* Hit@1 doubled after reranking.
* Mean Reciprocal Rank improved from 0.60 to 0.80.
* Queries whose correct document was already present in the candidate set were consistently promoted to the top position.
* Reranking cannot recover documents that are absent from the initial retrieval candidate set, highlighting the distinction between retrieval recall and ranking precision.

---

# Repository Structure

```text
src/
    app.py
    graph.py
    retrieve.py
    rerank.py
    ingest.py

prompts/
sources/
data/
tests/
```

---

# Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```text
OPENAI_API_KEY=
PINECONE_API_KEY=
COHERE_API_KEY=
```

Run the application:

```bash
python -m streamlit run src/app.py
```

---

# Future Work

* Regulatory monitoring using external APIs
* Learning history visualization
* Adaptive learning recommendations
* Retrieval quality dashboard
* Expanded regulatory corpus
* Optional hybrid retrieval (semantic + keyword)
* Production deployment

---

# Acknowledgements

Developed as a capstone project for the **Ironhack AI Engineering Bootcamp**, demonstrating Retrieval-Augmented Generation, semantic search, neural reranking, structured prompting, evaluation-driven development, and regulatory AI applications in drug development.

# DrugDev-AI

**DrugDev-AI** is an AI-powered assistant for pharmaceutical drug development and regulatory science. It combines Retrieval-Augmented Generation (RAG), structured learning, semantic search, neural reranking, and live regulatory monitoring into a single application.

Developed as the project for the **Ironhack AI Consulting Bootcamp**, the project demonstrates how modern AI workflows can support professionals and students throughout the drug development lifecycle. :contentReference[oaicite:0]{index=0}

---

# Highlights

- Three integrated workflows:
  - **Ask** — grounded regulatory Q&A
  - **Learn** — structured AI-assisted learning
  - **Monitor** — live regulatory intelligence
- Retrieval-Augmented Generation using authoritative regulatory documents
- Two-stage retrieval with semantic search and neural reranking
- AI-generated lessons and objective-based quizzes
- Live monitoring from multiple regulatory sources
- End-to-end tested with `pytest`

---

# Application Modes

## Ask Mode

Ask natural-language questions about pharmaceutical development, clinical trials, quality systems, pharmacovigilance, regulatory affairs, and related topics.

Features:

- Retrieval-Augmented Generation (RAG)
- Semantic retrieval using Pinecone
- Neural reranking using Cohere
- Grounded responses using the OpenAI Responses API
- Source citations for transparency and traceability
- Persistent conversation state within the application

---

## Learn Mode

Learn Mode provides a structured curriculum covering the major areas of pharmaceutical development.

Current curriculum includes:

- Drug Development Fundamentals
- Regulatory Landscape
- Clinical Trials
- Marketing Authorization
- Pharmaceutical Quality Systems
- Pharmacovigilance
- Emerging Topics

Each module includes:

- AI-generated lessons grounded in regulatory documents
- Clearly defined learning objectives
- Module-specific Q&A
- Automatically generated quizzes
- Concept-based quiz evaluation
- Progress tracking
- Module progression

Lessons are generated first and quizzes are created directly from the generated lesson, ensuring assessment matches the taught material.

---

## Monitor Mode

Monitor Mode aggregates recent regulatory developments from multiple authoritative external sources.

Currently supported:

- ClinicalTrials.gov
- openFDA
- European Medicines Agency (EMA)

Features:

- Live retrieval from external APIs/RSS
- Unified normalized data model
- Cross-source orchestration
- Partial failure handling
- Source-specific filtering
- Local keyword filtering
- Newest-first ranking
- Deduplication across sources

Monitor is intentionally independent from the RAG pipeline and retrieves live information directly from official sources.

---

# System Architecture

DrugDev-AI consists of three complementary workflows.

## Ask

```text
Question
      │
      ▼
OpenAI Embeddings
      │
      ▼
Pinecone Vector Database
      │
      ▼
Top-k Retrieval
      │
      ▼
Cohere Rerank
      │
      ▼
Grounded OpenAI Response
```

## Learn

```text
Curriculum Module
        │
        ▼
Retrieve Module Context
        │
        ▼
Generate Lesson
        │
        ▼
Generate Quiz
        │
        ▼
Evaluate Answers
        │
        ▼
Progress Tracking
```

## Monitor

```text
Topic
   │
   ▼
ClinicalTrials.gov
openFDA
EMA RSS
   │
   ▼
Normalization
   │
   ▼
Monitor Orchestrator
   │
   ▼
Unified Monitor Feed
```

---

# Knowledge Base

The RAG knowledge base contains regulatory and scientific documents from major international organizations, including:

- ICH
- FDA
- EMA
- WHO
- European Union
- WMA
- GVP

Topics include:

- Drug Development
- Clinical Trials
- Good Clinical Practice
- Pharmacovigilance
- Pharmaceutical Quality Systems
- Marketing Authorization
- Quality Risk Management
- Artificial Intelligence in Healthcare
- EU Pharmaceutical Regulation

The corpus currently contains approximately **6,600+ vectorized text chunks** stored in Pinecone. :contentReference[oaicite:1]{index=1}

---

# Technology Stack

## AI

- OpenAI Responses API
- OpenAI Embeddings
- Cohere Rerank

## Retrieval

- Pinecone Vector Database

## Live Monitoring

- ClinicalTrials.gov API v2
- openFDA API
- EMA RSS feeds

## Application

- Python
- Streamlit
- PyPDF
- YAML

## Testing

- pytest

---

# Retrieval Pipeline

DrugDev-AI uses a two-stage retrieval pipeline.

1. Semantic retrieval from Pinecone
2. Neural reranking with Cohere
3. Grounded answer generation with OpenAI

Separating retrieval from reranking improves ranking precision while maintaining semantic recall.

---

# Retrieval Evaluation

Retrieval quality was evaluated before and after introducing neural reranking.

| Metric | Baseline | Reranked |
|---------|---------:|---------:|
| Hit@1 | 0.40 | **0.80** |
| Hit@3 | 0.80 | 0.80 |
| Hit@5 | 0.80 | 0.80 |
| Mean Reciprocal Rank | 0.60 | **0.80** |

Key observations:

- Hit@1 doubled after reranking.
- Mean Reciprocal Rank increased substantially.
- Relevant documents are promoted to the top of the retrieved context before answer generation.

---

# Repository Structure

```text
src/
    app.py
    graph.py
    retrieve.py
    rerank.py
    ingest.py
    quiz.py
    learning.py
    monitor.py
    monitor_sources/
        clinical_trials.py
        openfda.py
        ema.py

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
OPENFDA_API_KEY=    # optional
```

Run:

```bash
streamlit run src/app.py
```

---

# Testing

Run the complete test suite:

```bash
pytest
```

or execute individual component suites, for example:

```bash
pytest tests/test_monitor*.py -v
pytest tests/test_quiz.py -v
pytest tests/test_learning.py -v
```

---

# Future Work

Planned improvements include:

- AI-generated Monitor summaries
- Saved Monitor searches
- Watchlists and notifications
- Cost and token usage dashboard
- Adaptive learning recommendations
- Learning analytics
- Hybrid retrieval (semantic + keyword)
- Production deployment

---

# Acknowledgements

DrugDev-AI was developed as the capstone project for the **Ironhack AI Consulting Bootcamp**. It demonstrates modern AI application design through Retrieval-Augmented Generation, neural reranking, structured prompting, evaluation-driven development, curriculum generation, and live regulatory monitoring in the pharmaceutical domain.
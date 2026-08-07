# Implementation Plan

## Project

AI Learning & Intelligence Assistant for Drug Development & Regulatory Science

---

# Objective

Implement the project incrementally, one module at a time.

Each module must:

- compile
- be independently testable
- pass all tests
- be committed before continuing

No future functionality should be implemented before the current module is complete.

---

# Overall Architecture

```
Official Documents
        ↓
Document Ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Pinecone
        ↓
Retriever
        ↓
LangGraph
        ↓
Ask | Learn | Monitor
        ↓
Streamlit
```

---

# Repository

```
project/

AGENT.md
project_plan.md
implementation_plan.md
curriculum.yaml

src/
sources/
data/
tests/
prompts/
```

---

# Phase 1 — Project Setup

Status: ✅ Completed

## Tasks

- [x] Repository created
- [x] Git initialized
- [x] .gitignore
- [x] .env
- [x] requirements.txt
- [x] OpenAI configuration
- [x] Pinecone configuration
- [x] Repository structure
- [x] curriculum.yaml
- [x] Source documents collected

Deliverable
Working OpenAI client
Working Pinecone client
Project structure established

---

# Phase 2 — Document Ingestion

Status: ⬜ Not Started

## Goal

Read documents from the sources directory.
Extract text.
Split into chunks.
Store document metadata.
No embeddings yet.

## Module

src/ingest.py

## Input

PDF documents

## Output

Document objects
Chunk objects
Metadata

## Test

tests/test_ingest.py

Verify

- documents are discovered
- PDFs load correctly
- text extraction works
- chunking works
- metadata generated

Deliverable
Working ingestion pipeline

---

# Phase 3 — Embeddings

Status: ⬜

## Goal

Generate embeddings for every chunk.

## Module

src/embeddings.py

## Input

Chunks

## Output

Vectors

## Test

tests/test_embeddings.py

Verify

- embeddings generated
- dimensions correct
- failures handled

Deliverable

Embedding pipeline

---

# Phase 4 — Pinecone Indexing

Status: ⬜

## Goal

Upload embeddings.
Store metadata.

## Module

src/embeddings.py

## Test

tests/test_pinecone.py

Verify

- upload successful
- metadata stored
- retrieval possible

Deliverable
Searchable vector database

---

# Phase 5 — Retrieval

Status: ⬜

## Goal

Semantic search.
Return relevant chunks.

## Module

src/retrieve.py

## Test

tests/test_retrieval.py

Verify

- relevant chunks returned
- similarity ordering
- metadata preserved

Deliverable
Retriever

---

# Phase 6 — Curriculum

Status: ⬜

## Goal

Load curriculum.yaml.
Represent curriculum internally.
Resolve prerequisites.
Retrieve associated documents.

## Module

src/curriculum.py

## Input

curriculum.yaml

## Output

Curriculum object

## Test

tests/test_curriculum.py

Verify

- YAML loads
- prerequisites resolved
- modules accessible
- document mappings correct

Deliverable
Working curriculum engine

---

# Phase 7 — LangGraph

Status: ⬜

## Goal

Build graph state.
Implement reusable nodes.

## Nodes

Retrieve
Generate
Respond
Later
Learn
Quiz
Monitor

## Module

src/graph.py

## Test

tests/test_graph.py
Deliverable
Working graph

---

# Phase 8 — Ask Mode

Status: ⬜

## Goal

General regulatory chatbot.

## Flow

Question
↓
Retrieve
↓
Generate
↓
Citation

## Module

src/chatbot.py

## Test

tests/test_chatbot.py

Verify

- grounded answer
- citations
- retrieval before generation

Deliverable
Ask mode complete

---

# Phase 9 — Learn Mode

Status: ⬜

## Goal

Guided learning.

## Components

Onboarding
Curriculum navigation
Learning session
Question answering
Quiz generation
Recommendation

## Modules

src/curriculum.py
src/quiz.py

## Test

tests/test_learning.py

Verify

- onboarding
- current module
- next recommendation
- quiz generation

Deliverable
Learn mode complete

---

# Phase 10 — Monitor Mode

Status: ⬜

## Goal

Summarize regulatory updates.
For MVP
Manual trigger.
Future
n8n automation.

## Module

src/monitor.py

## Test

tests/test_monitor.py
Deliverable
Monitor mode complete

---

# Phase 11 — Streamlit

Status: ⬜

## Goal

Implement user interface.
Tabs
Ask
Learn
Monitor

## Module

src/app.py

## Test

Manual
Deliverable
Working application

---

# Phase 12 — Integration

Status: ⬜

## Goal

Connect every module.
Perform end-to-end testing.
Refine prompts.
Improve logging.
Deliverable
Complete MVP

---

# Final Deliverables

- Working application
- GitHub repository
- README
- project_plan.md
- implementation_plan.md
- AGENT.md
- curriculum.yaml
- lab_proof.md

---

# Development Rules

Every implementation step follows exactly this sequence.

1.

Read AGENT.md
Read implementation_plan.md

2.

Implement only the current module.

3.

Explain briefly

- purpose
- inputs
- outputs

4.

Create or update tests.

5.

Run tests.

6.

If tests fail
fix only that module.

7.

If tests pass
update this document.

8.

Commit.

9.

Push.

10.

Only then continue.

---

# Commit Strategy

One completed module = one commit.
Examples

```
Configure OpenAI and Pinecone
Implement PDF ingestion
Implement chunk generation
Implement embeddings
Implement retrieval
Implement curriculum loader
Implement LangGraph workflow
Implement Ask mode
Implement Learn mode
Implement Monitor mode
Build Streamlit interface
```

---

# Definition of Done

A module is complete only when:

- implementation finished
- independently testable
- tests pass
- documentation updated
- committed
- pushed
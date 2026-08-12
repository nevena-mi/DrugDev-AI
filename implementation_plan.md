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

Status: ✅ Completed

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

Status: ✅ Completed

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

Status: ✅ Completed

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

Status: ✅ Completed

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

# Phase 6 — LangGraph / RAG Generation

Status: ✅ Completed

## Goal

Build the first complete RAG workflow for the Ask mode.

The workflow must take a natural-language user question, retrieve relevant document chunks, generate a grounded answer from the retrieved context, and return source citations.

## Flow

Question
↓
Retrieve relevant chunks
↓
Build grounded context
↓
Generate answer with LLM
↓
Return answer with citations

## Modules

src/graph.py
src/chatbot.py

## Input

Natural-language user question.

## Output

Grounded answer containing:

* answer text
* citations
* retrieved source metadata

## LangGraph Nodes

Retrieve
Generate
Respond

Do not implement Learn or Monitor nodes yet.

## LLM Rules

* Use the configured chat model.
* During development use GPT-4o-mini.
* Generate answers only from retrieved context.
* Do not answer from general model knowledge when the retrieved documents do not support the answer.
* Clearly state when the available knowledge base does not contain sufficient information.
* Preserve source metadata for citations.

## Tests

tests/test_graph.py
tests/test_chatbot.py

Verify:

* retrieval occurs before generation
* retrieved chunks are passed to the generation node
* grounded answer is produced
* citations are returned
* insufficient-context cases are handled safely
* mocked LLM calls are used in normal unit tests

## Deliverable

Working Ask-mode RAG backend.

---

# Phase 7 — Streamlit UI

Status: ✅ Completed

## Goal

Create the first usable interface for the application.

For this phase, implement the complete Ask interface and prepare the UI structure for the future Learn and Monitor modes.

## Module

src/app.py

## UI Structure

Ask
Learn
Monitor

At this stage:

* Ask is functional
* Learn may display a placeholder
* Monitor may display a placeholder

## Ask Mode

The user can:

* enter a natural-language question
* submit the question
* receive a grounded answer
* inspect citations and source information

## Rules

The UI must contain no retrieval or LLM business logic.

All processing must call backend functions from the existing modules.

## Test

Manual UI test.

Verify:

* application launches
* Ask mode accepts input
* Ask mode calls the backend
* answer is displayed
* citations are displayed
* errors are handled without crashing the application

## Deliverable

Working Streamlit application with functional Ask mode and placeholders for Learn and Monitor.

---

# Phase 8 — End-to-End Testing

Status: ⬜

## Goal

Verify that the complete Ask pipeline works with real project data.

## Flow Under Test

PDF documents
↓
Ingestion
↓
Chunking
↓
Embeddings
↓
Pinecone indexing
↓
User question
↓
Query embedding
↓
Retrieval
↓
LangGraph
↓
LLM generation
↓
Citation
↓
Streamlit output

## Evaluation Queries

Use representative questions covering:

* drug development
* clinical trials
* regulatory science
* pharmacovigilance
* pharmaceutical quality systems

## Verify

* documents are correctly indexed
* relevant chunks are retrieved
* answers are grounded in retrieved content
* citations identify the correct source documents
* unsupported questions are handled safely
* the application performs the complete path without manual intervention between steps

## Output

Record representative end-to-end examples for later retrieval evaluation and lab proof.

## Deliverable

Verified Ask-mode MVP.

---

# Phase 9 — Retrieval Quality Evaluation

Status: ⬜

## Prerequisite

Phase 8 end-to-end Ask workflow must be complete and working.

## Goal

Measure the quality of semantic retrieval before deciding whether reranking is necessary.

## Evaluation Process

Create a small representative evaluation set covering:

* drug development
* clinical trials
* regulatory science
* pharmacovigilance
* quality systems

For each question:

* inspect the top-k retrieved chunks
* determine whether the most relevant source appears
* record ordering and similarity scores
* identify irrelevant or redundant retrieved chunks
* record whether retrieval quality affects the final answer

## Output

Create a documented retrieval baseline.

Suggested file:

data/evaluation/retrieval_baseline.md

## Decision

After reviewing the baseline:

* If retrieval quality is sufficient, do not implement reranking.
* If retrieval quality is consistently insufficient, proceed to Phase 9b.

The decision to implement reranking must be made explicitly after reviewing the evaluation results.

## Deliverable

Documented semantic-retrieval baseline and reranking decision.

---

# Phase 9b — Optional Reranking

Status: ⬜ Conditional

Retrieval baseline established; reranking decision deferred until corpus expansion and Learn-mode evaluation.

## Prerequisite

Phase 9 must demonstrate that semantic retrieval alone is insufficient.

## Goal

Improve relevance by applying a second-stage reranker to the candidates returned by Pinecone.

## Module

src/rerank.py

## Test

tests/test_rerank.py

## Input

Candidate chunks returned by src/retrieve.py.

## Output

The same candidate chunks reordered by relevance, with only the highest-ranked chunks forwarded to generation.

## Flow

User question
↓
Pinecone semantic retrieval
↓
Candidate chunks
↓
Reranker
↓
Best context chunks
↓
LLM generation

## Evaluation

Use exactly the same query set as the Phase 9 baseline.

Compare:

* retrieval-only results
* retrieval + reranking results

Evaluate whether reranking improves relevance enough to justify its additional latency, complexity, and cost.

## Implementation Rule

Keep reranking independent from retrieval so it can be enabled, disabled, or replaced without changing the retriever.

## Deliverable

If justified, a tested independent reranking module and documented before/after comparison.

---



# Phase 10 — Learn Mode

Status: ⬜

## Goal

Implement guided learning using the existing curriculum.yaml and RAG infrastructure.

## Modules

src/curriculum.py
src/quiz.py

Additional learning-specific LangGraph nodes may be added to src/graph.py.

## Curriculum Source

curriculum.yaml

The curriculum must never be hardcoded into Python.

## Curriculum Engine

Load:

* modules
* titles
* descriptions
* prerequisites
* learning objectives
* associated documents
* difficulty
* duration
* quiz configuration

Resolve prerequisite relationships and determine valid learning progression.

## Learner Onboarding

Collect basic information such as:

* professional or academic background
* familiarity with drug development
* learning goal
* prior regulatory experience

Use the information to recommend an appropriate starting module.

The LLM may recommend an entry point but must select from the predefined curriculum.

## Learning Workflow

Onboarding
↓
Determine entry module
↓
Retrieve module documents
↓
Explain learning objective
↓
User asks free-form questions
↓
Ground answers in RAG sources
↓
Quiz
↓
Evaluate understanding
↓
Recommend next curriculum module

## Free-Form Questions

The learner may ask questions at any point.

Questions are processed through the same retrieval and grounded-generation infrastructure used by Ask mode.

The curriculum provides navigation but does not restrict user questions.

## Quiz

src/quiz.py

Generate short knowledge checks using retrieved curriculum context.

Do not evaluate answers solely from general model knowledge.

## Tests

tests/test_curriculum.py
tests/test_learning.py
tests/test_quiz.py

Verify:

* YAML loads correctly
* modules are accessible
* prerequisites resolve correctly
* document mappings are correct
* onboarding recommends a valid curriculum module
* free-form questions work during learning
* quiz generation is grounded in curriculum content
* next-module recommendations respect prerequisites

## Streamlit

Replace the Learn placeholder with:

* curriculum overview
* current module
* learning content
* question interface
* quiz
* progress / next-module recommendation

## Deliverable

Working Learn mode.

---
# Phase 11 — Monitor Mode

Status: ⬜

## Goal

Implement a live Regulatory Intelligence workspace that complements the existing **Ask** and **Learn** modes by providing recent developments from official regulatory sources.

Unlike Ask and Learn, which operate primarily on the curated RAG knowledge base, Monitor is **API-first** and focuses on current regulatory activity. Retrieved live information may subsequently be summarized and explained using the existing LLM infrastructure.

For the MVP, Monitor is manually triggered through the Streamlit interface. Automated scheduled monitoring remains a future extension.

---

## Module

`src/monitor.py`

Additional source adapters:

- `src/monitor_sources/openfda.py`
- `src/monitor_sources/clinical_trials.py`
- `src/monitor_sources/ema.py`

---

## Initial Scope

Integrate three official live information sources:

- ClinicalTrials.gov
- openFDA
- EMA RSS (or equivalent official EMA update feed)

Each source should be implemented independently behind a common interface so additional agencies can be added without changing the Monitor workflow.

---

## Data Model

Normalize all external results into a common `MonitorItem` model containing:

- source
- title
- publication date
- category/type
- short description
- official source URL
- source-specific identifier

The Streamlit UI must consume only the normalized model and remain independent of individual APIs.

---

## Workflow

User enters:
- topic or keyword
- selected sources
- optional recent time window
↓
Retrieve updates from selected official APIs
↓
Normalize into `MonitorItem`
↓
Sort chronologically
↓
(Optional) Filter locally without additional API calls
↓
Generate grounded AI summary explaining:
- **What changed?**
- **Why it may matter?**
↓
Display results with links to the official sources.

---

## AI Summaries

The LLM should summarize only the retrieved API content.

It must:

- avoid outside knowledge;
- clearly distinguish facts from interpretation;
- explain significance only when supported by the available information;
- state when insufficient information is available.

---

## Streamlit UI

Replace the Monitor placeholder with a functional interface containing:

- topic/keyword search
- source selection
- optional time-window filter
- **Fetch Updates** button
- chronological signal feed
- grounded AI summary
- links to official source documents

The interface should prioritize concise regulatory signals rather than raw API responses.

---

## Integration

Monitor should reuse the existing project infrastructure where appropriate:

- OpenAI client
- prompt loading
- structured outputs
- logging
- error handling

It should **not** depend on Pinecone retrieval for current updates, although future versions may ingest retrieved updates into the knowledge base.

A future enhancement may allow users to send a selected Monitor item directly to Ask mode for deeper explanation.

---

## Tests

`tests/test_monitor.py`

Verify:

- API adapters normalize responses correctly
- `MonitorItem` objects are constructed correctly
- keyword filtering works
- chronological ordering is correct
- partial API failures do not stop the workflow
- AI summaries remain grounded in retrieved API content
- official source URLs are preserved
- mocked APIs are used during unit tests

---

## Future Extension

Future versions may introduce:

- scheduled monitoring via n8n
- automatic ingestion of new documents into the RAG corpus
- daily or weekly summaries
- email or Slack notifications
- saved searches
- watchlists
- regulatory impact scoring
- historical trend analysis
- semantic search across stored regulatory updates

---

## Deliverable

A working **Monitor** mode that retrieves live updates from official regulatory sources, presents them as normalized regulatory signals, explains their significance through grounded AI summaries, and integrates cleanly with the existing **Ask** and **Learn** architecture.
---

# Phase 12 — Final Integration & MVP Validation

Status: ⬜

## Goal

Validate the complete three-mode application.

## Modes

Ask
Learn
Monitor

## Verify

* shared RAG infrastructure works across all modes
* citations are consistently returned
* curriculum navigation works
* Monitor mode produces grounded summaries
* Streamlit routes requests to the correct workflow
* errors do not break the application
* no mode duplicates retrieval or embedding logic unnecessarily

## Final Deliverable

Complete MVP of the AI Learning & Intelligence Assistant for Drug Development & Regulatory Science.

---

# Final Deliverables

* Working three-mode application
* Ask mode
* Learn mode
* Monitor mode
* GitHub repository
* README.md
* project_plan.md
* implementation_plan.md
* AGENT.md
* curriculum.yaml
* retrieval evaluation
* lab_proof.md

---

# Development Rules

Every implementation step follows exactly this sequence.

1. Read AGENT.md and implementation_plan.md.

2. Implement only the current phase.

3. Explain briefly:

   * purpose
   * input
   * output
   * connection to existing modules

4. Create or update tests.

5. Run tests.

6. If tests fail, fix only the current phase.

7. If tests pass, update implementation_plan.md.

8. User decides when to commit and push.

9. Only then continue to the next phase.

Do not perform Git commits or pushes automatically.

---

# Commit Strategy

One completed and tested phase should correspond to one focused commit.

Examples:

```text
Configure OpenAI and Pinecone
Implement PDF ingestion
Implement embedding pipeline
Implement Pinecone indexing
Implement semantic retrieval
Implement RAG generation workflow
Build Ask interface
Validate end-to-end Ask pipeline
Evaluate retrieval quality
Add reranking if justified
Implement Monitor mode
Implement Learn mode
Complete application integration
```

---

# Definition of Done

A phase is complete only when:

* implementation is finished
* it is independently testable where applicable
* relevant tests pass
* integration with previous phases is verified
* implementation_plan.md is updated
* the user has reviewed the result

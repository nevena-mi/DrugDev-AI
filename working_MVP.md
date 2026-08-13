# Project 3 Deliverables — DrugDev-AI

## 1. Working MVP

**Primary stack:** LangGraph

DrugDev-AI is implemented as a runnable Python/Streamlit application.

The MVP contains three functional modes:

### Ask
Grounded regulatory question answering using:

OpenAI embeddings  
→ Pinecone semantic retrieval  
→ Cohere reranking  
→ OpenAI grounded answer generation  
→ document citations

User Question
↓
OpenAI Embedding
↓
Pinecone Semantic Retrieval — Top 15
↓
Cohere Reranking
↓
Top 5 Relevant Chunks
↓
Context Assembly
↓
OpenAI Grounded Answer Generation
↓
Citations

### Learn
Structured learning workflow including:

module selection  
→ module-scoped retrieval and reranking  
→ grounded lesson generation  
→ quiz generation  
→ quiz evaluation  
→ module progression

Learner Profile
↓
Starting Module Selection
↓
Module-Scoped Retrieval — Top 15
↓
Cohere Reranking
↓
Top 5 Grounding Chunks
↓
Lesson Generation
↓
Module Q&A
↓
Quiz Generation
↓
Learner Answers
↓
Quiz Evaluation
↓
Module Progression

### Monitor
Live regulatory monitoring using:

ClinicalTrials.gov  
+ openFDA  
+ EMA RSS  
→ normalization into `MonitorItem`  
→ Monitor orchestrator  
→ filtering, sorting and unified signal feed

Application entrypoint:

```bash
python -m streamlit run src/app.py
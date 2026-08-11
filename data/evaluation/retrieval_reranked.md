# Retrieval Reranked Report

## Summary

- Corpus size: 37 PDFs
- Number of evaluation queries: 5
- Mean Hit@1: 0.8000
- Mean Hit@3: 0.8000
- Mean Hit@5: 0.8000
- Mean Reciprocal Rank (MRR): 0.8000

## Comparison Table

| Metric | Baseline | Reranked | Change |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.4000 | 0.8000 | +0.4000 |
| Hit@3 | 0.8000 | 0.8000 | +0.0000 |
| Hit@5 | 0.8000 | 0.8000 | +0.0000 |
| MRR | 0.6000 | 0.8000 | +0.2000 |

## Per-Query Results

### `gcp_definition`
- Question: What is Good Clinical Practice?
- Expected primary source: ICH E6(R3) Guideline for Good Clinical Practice
- Final rank in top 5: 1
- Rank in original top 15: 2
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 reranked titles and scores:
  1. ICH E6(R3) Guideline for Good Clinical Practice — Pinecone: 0.6803 | Cohere: 0.9350
  2. Good Clinical Laboratory Practice (GCLP) — Pinecone: 0.6901 | Cohere: 0.9159
  3. FDA Drug Development and Approval Process — Pinecone: 0.6318 | Cohere: 0.8923
  4. ICH E6(R3) Guideline for Good Clinical Practice — Pinecone: 0.6415 | Cohere: 0.8411
  5. FDA Drug Development and Approval Process — Pinecone: 0.5983 | Cohere: 0.7770

### `quality_risk_management`
- Question: What is Quality Risk Management and what are its main principles?
- Expected primary source: ICH Q9 Quality Risk Management
- Final rank in top 5: 1
- Rank in original top 15: 2
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 reranked titles and scores:
  1. ICH Q9 Quality Risk Management — Pinecone: 0.6670 | Cohere: 0.9588
  2. ICH Q9 Quality Risk Management — Pinecone: 0.6468 | Cohere: 0.9087
  3. ICH Q9 Quality Risk Management — Pinecone: 0.6330 | Cohere: 0.9004
  4. ICH Q10 Pharmaceutical Quality System — Pinecone: 0.6805 | Cohere: 0.8815
  5. ICH Q9 Quality Risk Management — Pinecone: 0.6352 | Cohere: 0.8689

### `pharmacovigilance_planning`
- Question: What is pharmacovigilance planning?
- Expected primary source: ICH E2E Pharmacovigilance Planning
- Final rank in top 5: 1
- Rank in original top 15: 1
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 reranked titles and scores:
  1. ICH E2E Pharmacovigilance Planning — Pinecone: 0.7607 | Cohere: 0.9728
  2. ICH E2E Pharmacovigilance Planning — Pinecone: 0.7209 | Cohere: 0.9602
  3. ICH E2E Pharmacovigilance Planning — Pinecone: 0.7058 | Cohere: 0.9419
  4. Good Pharmacovigilance Practices Module V — Pinecone: 0.6879 | Cohere: 0.9381
  5. ICH E2E Pharmacovigilance Planning — Pinecone: 0.6845 | Cohere: 0.9275

### `ind`
- Question: What is the IND application?
- Expected primary source: Investigational New Drug (IND) Application
- Final rank in top 5: 1
- Rank in original top 15: 1
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 reranked titles and scores:
  1. Investigational New Drug (IND) Application — Pinecone: 0.7507 | Cohere: 0.8777
  2. Investigational New Drug (IND) Application — Pinecone: 0.7458 | Cohere: 0.8686
  3. Investigational New Drug (IND) Application — Pinecone: 0.6977 | Cohere: 0.8503
  4. Investigational New Drug (IND) Application — Pinecone: 0.7353 | Cohere: 0.8406
  5. Investigational New Drug (IND) Application — Pinecone: 0.7309 | Cohere: 0.8363

### `eu_ai_act`
- Question: How does the EU AI Act affect AI systems used in healthcare?
- Expected primary source: Regulation (EU) 2024/1689 Artificial Intelligence Act
- Final rank in top 5: not found
- Rank in original top 15: not found
- Hit@1: 0
- Hit@3: 0
- Hit@5: 0
- Reciprocal rank: 0.0000
- Top-5 reranked titles and scores:
  1. Ethics and Governance of Artificial Intelligence for Health — Pinecone: 0.6527 | Cohere: 0.6906
  2. Ethics and Governance of Artificial Intelligence for Health — Pinecone: 0.6450 | Cohere: 0.6633
  3. Ethics and Governance of Artificial Intelligence for Health — Pinecone: 0.6470 | Cohere: 0.6510
  4. Ethics and Governance of Artificial Intelligence for Health — Pinecone: 0.6446 | Cohere: 0.6276
  5. Ethics and Governance of Artificial Intelligence for Health — Pinecone: 0.6651 | Cohere: 0.6184

## Observations

- Queries not found in the final top 5: eu_ai_act
- Queries that reranking cannot recover from the original top 15: eu_ai_act (not present in original top 15)

## Decision Analysis

- Reranking is justified for production consideration.
- Hit@1 improved: yes.
- MRR improved: yes.
- Hit@3 stayed equal or improved: yes.
- Hit@5 stayed equal or improved: yes.
- GCP moved above GCLP: yes.
- ICH Q9 moved above ICH Q10: yes.
- EU AI Act was not present in the original top 15 candidates, so reranking cannot recover that miss.
- Added Cohere latency and API usage: one Cohere rerank call per evaluation query on top of the existing Pinecone retrieval call.

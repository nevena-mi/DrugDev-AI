# Retrieval Baseline Report

## Summary

- Corpus size: 37 PDFs
- Number of evaluation queries: 5
- Mean Hit@1: 0.4000
- Mean Hit@3: 0.8000
- Mean Hit@5: 0.8000
- Mean Reciprocal Rank (MRR): 0.6000

## Per-Query Results

### `gcp_definition`
- Question: What is Good Clinical Practice?
- Expected primary source: ICH E6(R3) Guideline for Good Clinical Practice
- Rank in top 5: 2
- Hit@1: 0
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 0.5000
- Top-5 retrieved document titles and scores:
  1. Good Clinical Laboratory Practice (GCLP) — 0.6901
  2. ICH E6(R3) Guideline for Good Clinical Practice — 0.6803
  3. Good Clinical Laboratory Practice (GCLP) — 0.6471
  4. ICH E6(R3) Guideline for Good Clinical Practice — 0.6415
  5. Good Clinical Laboratory Practice (GCLP) — 0.6399

### `quality_risk_management`
- Question: What is Quality Risk Management and what are its main principles?
- Expected primary source: ICH Q9 Quality Risk Management
- Rank in top 5: 2
- Hit@1: 0
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 0.5000
- Top-5 retrieved document titles and scores:
  1. ICH Q10 Pharmaceutical Quality System — 0.6805
  2. ICH Q9 Quality Risk Management — 0.6670
  3. ICH Q9 Quality Risk Management — 0.6659
  4. ICH Q9 Quality Risk Management — 0.6567
  5. ICH Q9 Quality Risk Management — 0.6497

### `pharmacovigilance_planning`
- Question: What is pharmacovigilance planning?
- Expected primary source: ICH E2E Pharmacovigilance Planning
- Rank in top 5: 1
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 retrieved document titles and scores:
  1. ICH E2E Pharmacovigilance Planning — 0.7607
  2. Good Pharmacovigilance Practices Module V — 0.7220
  3. ICH E2E Pharmacovigilance Planning — 0.7209
  4. ICH E2E Pharmacovigilance Planning — 0.7134
  5. Good Pharmacovigilance Practices Module V — 0.7101

### `ind`
- Question: What is the IND application?
- Expected primary source: Investigational New Drug (IND) Application
- Rank in top 5: 1
- Hit@1: 1
- Hit@3: 1
- Hit@5: 1
- Reciprocal rank: 1.0000
- Top-5 retrieved document titles and scores:
  1. Investigational New Drug (IND) Application — 0.7507
  2. Investigational New Drug (IND) Application — 0.7458
  3. Investigational New Drug (IND) Application — 0.7353
  4. Investigational New Drug (IND) Application — 0.7309
  5. Investigational New Drug (IND) Application — 0.7290

### `eu_ai_act`
- Question: How does the EU AI Act affect AI systems used in healthcare?
- Expected primary source: Regulation (EU) 2024/1689 Artificial Intelligence Act
- Rank in top 5: not found
- Hit@1: 0
- Hit@3: 0
- Hit@5: 0
- Reciprocal rank: 0.0000
- Top-5 retrieved document titles and scores:
  1. Ethics and Governance of Artificial Intelligence for Health — 0.6665
  2. Ethics and Governance of Artificial Intelligence for Health — 0.6651
  3. Ethics and Governance of Artificial Intelligence for Health — 0.6557
  4. Ethics and Governance of Artificial Intelligence for Health — 0.6540
  5. Ethics and Governance of Artificial Intelligence for Health — 0.6527

## Observations

- Queries not found in the top 5: eu_ai_act
- Queries where the expected source was not ranked first: gcp_definition (rank 2), quality_risk_management (rank 2)

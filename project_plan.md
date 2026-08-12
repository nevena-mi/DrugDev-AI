### An AI-powered learning and regulatory intelligence assistant for pharmaceutical professionals.


# AI Learning & Intelligence Assistant for Drug Development & Regulatory Science

**Version:** 1.0  
**Author:** Nevena Milenkovic Zujko  
**Project Type:** Autonomous AI Agent (LangGraph + RAG)  
**Target:** Ironhack Autonomous Agents Project & GitHub Portfolio

---

# 1. Executive Summary

Drug development and regulatory science require professionals to navigate an enormous and continuously evolving body of official documentation. Guidance from agencies such as the European Medicines Agency (EMA), the U.S. Food and Drug Administration (FDA), the International Council for Harmonisation (ICH), and the World Health Organization (WHO) is essential for professionals working in pharmaceutical research, clinical development, regulatory affairs, quality, pharmacovigilance, and medical affairs.

Although authoritative information is publicly available, it is fragmented across multiple organizations, difficult to search efficiently, and often challenging for newcomers to understand. Existing enterprise Regulatory Intelligence platforms primarily focus on document management and monitoring, while educational platforms provide static courses without interactive, personalized learning.

The proposed project combines both worlds into a single AI-powered assistant that supports learning, question answering, and regulatory intelligence through retrieval-augmented generation (RAG).

Instead of functioning as another chatbot or newsletter aggregator, the assistant acts as an intelligent guide through the complete drug development and regulatory landscape.

---

# 2. Existing Solutions

Several commercial and public products already address parts of this problem.

## Enterprise Regulatory Intelligence

- Cortellis Regulatory Intelligence
- Clarivate Regulatory Intelligence
- Veeva Development Cloud
- IQVIA Regulatory Solutions
- PharmaLex Consulting

These systems provide excellent monitoring and document management but are expensive enterprise solutions and do not function as AI tutors.

## Educational Platforms

- FDA Learning Portal
- CDRH Learn
- Regulatory Academy
- DIA educational resources

These provide structured educational content but lack conversational AI, semantic search and personalized learning.

## General LLMs

ChatGPT, Claude and Gemini can answer regulatory questions but have no curated regulatory knowledge base, no structured curriculum, no learning progression and no guarantee that answers are grounded in official documentation.

---

# 3. Project Vision

Create an AI assistant that combines

- Regulatory Intelligence
- Interactive Learning
- Expert Question Answering

using only official regulatory documentation.

The system should become an intelligent mentor rather than simply a search engine.

---

# 4. Target Users

## Primary Users

- Scientists transitioning into Pharma
- Regulatory Affairs Specialists
- Clinical Research Associates
- Clinical Operations Professionals
- Medical Affairs
- Pharmacovigilance Professionals
- Quality Assurance Professionals

## Secondary Users

- Biomedical PhD students
- Life Science students
- Pharmaceutical companies
- MedTech companies
- Regulatory consultants

---

# 5. Problems Addressed

Current workflows require users to

- search multiple regulatory websites
- subscribe to newsletters
- manually compare updated guidance
- search hundreds of pages of PDFs
- understand unfamiliar terminology
- decide what to learn first

The proposed assistant reduces this effort through semantic retrieval, AI explanations and guided learning.

---

# 6. Competitive Advantage

Unlike existing products, this assistant combines three capabilities within one system.

## Ask

An AI assistant capable of answering free-form questions using only retrieved official documents.

## Learn

A personalized learning environment that guides users through Drug Development and Regulatory Science using a predefined curriculum while adapting explanations and pacing to the learner.

## Monitor

Continuous monitoring of regulatory agencies, summarizing updates and explaining why changes matter.

The same RAG knowledge base powers all three modes.

---

# 7. Use Cases

### Ask

"What is Good Clinical Practice?"

"What is the difference between EMA and FDA approval?"

"Summarize ICH E6."

---

### Learn

"I am a neuroscientist moving into Regulatory Affairs."

"Create a learning path."

"Quiz me on Clinical Trials."

---

### Monitor

"What changed this week?"

"Has EMA updated pharmacovigilance guidance?"

"Summarize the newest FDA announcement."

---

# 8. User Interface

The Streamlit application contains three primary modes.

Ask | Learn | Monitor


## Ask

General chatbot

Natural language questions

Source citations

Conversation history

---

## Learn

Personalized onboarding

Curriculum visualization

Progress tracking

Learning recommendations

Quizzes

---

## Monitor

Latest regulatory updates

Agency filters

Change summaries

Future versions:

Email digest

Slack notifications

Teams integration

---

# 9. System Architecture
Official Sources
EMA
FDA
ICH
WHO
↓
Document Loader
↓
PDF Parsing
↓
Chunking
↓
Metadata Extraction
↓
Embeddings
↓
Pinecone
↓
LangGraph
↓
Ask
Learn
Monitor
↓
Streamlit UI


---

# 10. Technology Selection Framework

| Requirement | Solution |
|------------|----------|
| Large document collections | RAG |
| Official document retrieval | APIs / PDFs |
| Multi-step workflows | LangGraph |
| Semantic search | Pinecone |
| Automation | n8n |
| User interface | Streamlit |

---

# 11. Technology Stack

| Component | Technology |
|------------|------------|
| LLM | GPT-5.5 (GPT-4o-mini during development) |
| Framework | LangGraph |
| RAG | LangChain |
| Embeddings | text-embedding-3-small |
| Vector DB | Pinecone |
| UI | Streamlit |
| Automation | n8n |
| Language | Python |

---

# 12. Knowledge Base

The assistant uses only authoritative sources.

Examples include:
EMA
FDA
ICH
WHO
European Commission
Future versions
MDR
IVDR
EU AI Act

---

# 13. Curriculum

The curriculum follows the lifecycle of medicine development rather than agency structure.

## Module 1

Introduction to Drug Development

---

## Module 2

Regulatory Landscape:
EMA
FDA
ICH
WHO

---

## Module 3

Drug Discovery and Development

---

## Module 4

Clinical Trials
ICH E6
Ethics
Monitoring

---

## Module 5

Marketing Authorization
Submission pathways
Review process
Approvals

---

## Module 6

Quality Systems
GMP
GLP
GDP
GCP
GVP

---

## Module 7

Pharmacovigilance
Risk management
Signal detection
Safety reporting

---

## Module 8

Manufacturing and Quality
ICH Q series
CMC

---

## Module 9

Lifecycle Management
Variations
Renewals
Post-marketing

---

## Module 10

Emerging Topics
AI
Real World Evidence
ATMP
Medical Devices

---

# 14. Curriculum Implementation

The curriculum is **not generated by the LLM**.

Instead it is defined as a curriculum graph.

Each topic contains

- prerequisites
- learning objectives
- associated documents
- quizzes
- estimated duration
- difficulty

Example:
Drug Development
↓
Clinical Trials
↓
Good Clinical Practice
↓
Marketing Authorization
↓
Pharmacovigilance


Documents are tagged during ingestion using predefined topic labels.

The LLM selects among existing topics rather than inventing new ones.

---

# 15. User Onboarding

The assistant begins with a conversational assessment.

Example questions:
Background
Career goal
Experience
Available study time
Preferred learning style
Based on these responses the assistant selects the appropriate entry point within the curriculum.

---

# 16. Learning Workflow

User
↓
Onboarding
↓
Determine current level
↓
Select module
↓
Retrieve documents
↓
Explain concept
↓
Questions
↓
Quiz
↓
Evaluate
↓
Recommend next topic


---

# 17. Ask Workflow
Question
↓
Retrieve
↓
Generate grounded answer
↓
Return citations

---

# 18. Monitor Workflow
Monitor mode is manually triggered and API-first.

User enters topic / keyword and optional filters
↓
Query official live sources:
ClinicalTrials.gov
openFDA
EMA
↓
Normalize results into a common MonitorItem model
↓
Filter and sort recent regulatory signals
↓
Generate grounded AI summary when requested
↓
Display:
- what changed
- why it may matter
- source
- date
- official link

Monitor does not use Pinecone or the existing RAG retrieval pipeline for live updates.

Future versions may add:

Scheduled monitoring via n8n
↓
Detect new or changed regulatory information
↓
Store monitoring history
↓
Generate daily / weekly summaries
↓
Send alerts or notifications


---

# 19. MVP Scope

## Must Have

Project setup
PDF ingestion
Chunking
Embeddings
Pinecone
Semantic retrieval
Ask mode
Learn mode
Basic curriculum
Onboarding
Simple quizzes
Streamlit interface
Monitor mode

---

## Version 2

Multiple agencies
Automatic document ingestion
Progress tracking
Adaptive curriculum
Email digest

---

## Version 3

Daily monitoring
Slack
Teams
Dashboard
Learning analytics
Document comparison
Impact analysis
Multi-user support

---

# 20. Risk Assessment

| Risk | Mitigation |
|--------|------------|
| Hallucinations | RAG with mandatory citations |
| Incorrect retrieval | Better chunking and reranking |
| Large documents | Semantic chunking |
| API costs | Cache embeddings |
| Curriculum quality | Manual expert-designed curriculum |
| Website changes | Modular ingestion pipeline |
| Scope creep | Strict MVP |

---


# 21. Success Metrics

## Functional

Successful document ingestion
Accurate retrieval
Grounded answers
Working learning path
Working chatbot

---

## User

Meaningful explanations
Useful learning recommendations
Answers with citations
Response time under ten seconds

---

## Technical

Modular architecture
Reusable workflows
Easy extension to additional agencies

---

# 22. Future Roadmap

The project is intentionally designed as a scalable platform.

Future development includes automatic ingestion through n8n, adaptive learning based on learner performance, personalized study plans, certification, interview preparation, multilingual support, comparison of document versions, regulatory impact analysis, and enterprise integrations.

The long-term vision is to create an AI mentor that supports professionals throughout the entire lifecycle of drug development—from learning the fundamentals to staying current with evolving regulatory requirements.
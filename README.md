# CV Fit Signal (AI RAG System)
## Evidence-based CV–Job Requirement Matching System

> A RAG system that evaluates how well a CV satisfies job requirements by grounding decisions in explicit / implicit evidence from the CV.

## Why This Project Exists

- Traditional CV screening systems often rely on keyword matching and fail to capture implicit skills, contextual experience, and evidence quality.
- This project explores whether Retrieval-Augmented Generation (RAG) can improve CV evaluation by grounding decisions in explicit evidence retrieved from the candidate's CV.

## Current Status
Implemented:
- Modular inference pipeline
- LLM-based CV parser
- Component-aware retrieval
- Evidence-based evaluation
- Configurable system
- User-friendly report generation

In Progress:
- Logging and failure handling
- Observability improvements
- Retrieval debugging

Planned:
- FastAPI integration
- Dockerization
- Requirement weighting

## Current Architecture
Pipeline:
CV + Job Requirement Input
        ↓
LLM-based CV Parser
        ↓
Semantic CV Chunking
        ↓
Job Requirement Decomposition and Validation
        ↓
Embedding Generation
        ↓
FAISS Retrieval
        ↓
Evidence Preparation
        ↓
LLM-based Evaluation
        ↓
Structured Scoring Logic
        ↓
LLM-based Report Generation

## Example Output
```json
{
    "datetime": "'07-05-2026 18:54",
    "name": "Ardi Pratama",
    "report": [{
        "query": "Experience applying machine learning techniques to real problems",
        "score": 0.425,
        "reason": "Your experience in building a classification pipeline and performing preprocessing tasks showcases your application of machine learning techniques, which is a strong asset. However, it would be beneficial to connect this experience to a specific real-world problem to strengthen your case further."},
        {"query":"Familiarity with containerization tools (e.g., Docker)",
        "score":0.0,
        "reason":"Unfortunately, there is no evidence of your familiarity with containerization tools like Docker, which is a key requirement for this role. Gaining experience in this area would enhance your qualifications significantly"},]
    "final_score": 0.215}
}
```

## Key Design Decisions
- Combined component-level retrieval with full job requirement retrieval
- Split evaluation into:
  - Capability Level
  - Evidence Strength
  - Responsibility Level
- Final score is calculated outside the LLM for more consistent scoring
- Evaluation reasoning is grounded using retrieved CV evidence
- Built using a modular service-based pipeline structure


## Observed Challenges
- LLM decomposition can sometimes produce components without enough context
- Weak retrieval quality can lower evaluation accuracy
- Multiplication-based scoring can produce overly low scores
- Messy or inconsistent CV formats still create parsing issues
- Too much retrieval context can reduce evaluation quality

## Problem
Most CV–job matching systems rely on keyword-based filtering (e.g., matching “Python” or “Docker”).
This approach fails in some ways:
- Lack of context — Keywords ignore how skills are applied (e.g., “FastAPI” implies REST API experience but is not explicitly matched)
- Implicit requirements — Job descriptions often describe requirements indirectly, making exact keyword matching insufficient
- No explainability — Traditional systems return scores without showing why a candidate matches or not
- No prioritization control — There is no clear distinction between required and nice-to-have qualifications


## Objective
Build a system that:
- Evaluates each job requirement individually
- Retrieves supporting evidence from the CV
- Outputs a fit score and reasoning
- Identifies missing requirements explicitly
- Planned extension: requirement weighting (must-have vs nice-to-have)


## Development Progress
### Stage 1 — Evaluation System
Status: Completed

Implemented:
- Requirement-level evaluation
- Evidence retrieval pipeline
- Component-aware decomposition
- Structured scoring logic
- Explainable reasoning generation


### Stage 2 — Inference System
Status: Current

Implemented:
- Modular inference pipeline
- LLM-based structured CV parser
- Semantic CV chunking
- Service-oriented architecture
- LLM-generated recruiter-style reports

In Progress:
- Logging
- Failure handling
- Observability improvements


### Stage 3 — Robustness & Observability
Planned:
- Structured logging
- Failure tracing
- Retrieval debugging
- Async / concurrent evaluation


### Stage 4 — API Layer (FastAPI)
Planned:
- /predict endpoint
- /evaluate endpoint
- Request/response schemas


### Stage 5 — Containerization
Planned:
- Docker support
- Config-driven execution
- Reproducible runtime environment


### Stage 6 — Feature Expansion
Planned:
- Requirement weighting
- Improved reranking
- Advanced fit scoring
- Better ranking calibration

Author:
Jearim Jarden
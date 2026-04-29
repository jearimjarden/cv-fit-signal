# CV Fit Signal (AI RAG System)
# Evidence-based CV–Job Requirement Matching System

> A RAG system that evaluates how well a CV satisfies job requirements by grounding decisions in explicit evidence from the CV.

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

## System Overview

Pipeline:

Job Requirements → Chunking → Embedding → Vector Search (FAISS)  
→ Evidence Retrieval → Prompt Augmentation → LLM Evaluation → Structured Output (JSON)

---

## Example Output

```json
{
    "requirement": "Handle structured data storage and querying",
    "match": "medium",
    "score": 0.5,
    "reason": "The candidate has experience with data retrieval and processing, but there is no explicit mention of handling structured data storage and querying in their evidence (Evidence [1], [2])."
}
```
## Development Stages
### Stage 1 — Evaluation System (Current)
> Build a controlled RAG pipeline to evaluate different design choices.

Focus:
- Requirement → Retrieval → Decision pipeline
- Chunking strategy exploration
- Retrieval quality validation
- Prompt design for strict, evidence-based outputs
- Structured evaluation outputs (CSV/JSON) for analysis

Output:
Per-requirement results:
- requirement
- match (binary / probability)
- retrieved chunks
- selected evidence
- missing aspects

Goal:
- Understand which combination of retrieval + prompting produces the most reliable and explainable decisions through measurable outputs.

### Stage 2 — Inference System
> Convert the best-performing evaluation setup into a stable inference pipeline.

Focus:
- Use validated configurations (chunking, k, prompt)
- Standardize input/output format
- Refactor into modular, scaleable component
- Ensure clear separation of concern

Goal:
- A clean, modular system that can scale without rewriting core logic.

### Stage 3 — Robustness & Observability
> Make the system production-aware.

Focus:
- Logging (retrieval results, decisions, errors)
- Failure handling (empty retrieval, ambiguous matches)
- Debug visibility per requirement
- Asynchronous / parallel execution of requirement evaluation

Goal:
- A debuggable and efficient system where multiple requirement can be processed concurently with controlled latency.

### Stage 4 — API Layer (FastAPI)
Expose the system through a clean interface.

Focus:
- /predict → inference endpoint
- /evaluate → debugging/evaluation endpoint
- Request/response schema design

Goal:
- Decouple core logic from interface and enable external usage.

### Stage 5 — Containerization (Docker)
> Prepare for reproducibility and deployment.

Focus:
- Dockerized environment
- Config-driven execution
- Consistent runtime setup

Goal:
- Run the system reliably in any environment.

### Stage 6 — Feature Expansion
> Extend system capability beyond binary matching.

Focus:
- Requirement weighting (must-have vs nice-to-have)
- Fit scoring system
- Improved reasoning and ranking

Goal:
- Move from binary decisions → nuanced candidate evaluation.

Author:
Jearim Jarden
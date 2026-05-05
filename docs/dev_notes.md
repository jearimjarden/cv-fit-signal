# Developer Notes
- Contain each developer decision and reason for each commit
- Provide notes for developer 

## 3. CV-fit RAG System v1.0
### a. Job Requirement Parser and Chunking
Decision:
- Job Requirements input should be structured
- Job Requirements parsed using "\n" (new line)
- Parsed Job Requirement straight used as chunks for embedding

Reason:
- Job Requirements posting in real-world usually structured and could be parsed by new line
- Total Job Requirement usually short so we straight use parsed Job Requirement as chunk. If we combine Job Requirement, the semantic meaning of each Job Requirement will be bias because of pooling dillution 
- This a evaluation system so Job Requirement's chunks need to have a strong semantic meaning that can be compared to evidence (evidence from CV usually has indirect/implicit meaning)

Notes:
- We could implement OCR and parsing using strict LLM to get structured Job Requirement (Need to be strict so LLM would not change exact word from Job Requirement)
- Need normalization for each parsed text

### b. Regex CV Parser
Decision:
- CV Parsed using Regex for words ("Name", "Summary", "Skills", "Experience", "Education")
- Text that separated by Regex will parsed again using "\n" (new line)

Reason:
- CV usually has a semi structured that we can separate by using regex
- This is not recommend for long term (because of probability of messy CV input) but for RAG testing we will use this as CV parser

Notes:
- Implement text normalization before searching with Regex (e.g "Name" to "name", " name " to "name")
- We could implement OCR and parser using strict LLM to get structutured CV

### c. Custom CV Chunking
Decision:
- Augmented CV chunks with contextual labels (e.g., section titles like Experience, Skills, Projects) and associated subtitles when available.

    Example:
    CV: 
    "Experience:
    Backend Developer Intern
    - Built REST APIs using FastAPI for internal tools"

    CV Chunk:
    "Experience: Backend Developer Intern Built REST APIs using FastAPI for internal tools"

Reason:
- Preserved contextual labels (e.g., Experience, Skills, Projects) to improve semantic alignment with similarly structured job requirements.
- Strengthens downstream evaluation (e.g., evidence scoring and capability classification) by providing clearer intent and role framing.

Notes:
- This Chunking require prestructured semi structured CV, random CV structured will fails resulting in random or incorrect chunk

### d. Job Requirement Decomposer
Decision:
- Decompose (if able) each Job Requirement into components using LLM
- Semi strict prompt are used
    
    Example:
        Job Requirement:
            "Experience applying machine learning techniques to real problems"

        Components:
            - "Experience applying machine learning techniques"
            - "Applying machine learning to real problems"

Reason:
- Job requirements define the target evidence, so decomposing them into specific, granular components enables more precise matching and evaluation,
- Improved Evaluation scoring by decomposing Job Requirement into evaluatable components,
- Used an LLM to decompose job requirements, as their complexity and variability cannot be reliably handled with rule-based systems alone.
- Very strict prompt usually decompose Job Requirement per word not context, while loose prompt usually added concept outside the Job Requirement's context.

Limitation:
- Job Requirement that is short, combined level of proficiency and the skills sometimes return uncontextualized components
- Job Requirement with many verbs return uncontextual components

Observed Issues:
- A sentence with many verbs return uncontextual components
    
    Example:
        Job Requirement:
            Collaborate using modern development tools and workflows

        Components:
            - Collaborate
            - using modern development tools
            - using modern development workflows

- A sentence that combined level of proficiency and the skills

        Example:
        Job Requirement:
            Strong proficiency in Python

        Components:
            - Strong proficiency
            - in Python

### e. Decomposer LLM failure handling
Decision:
- Catch invalid components using rule-based system
- Generate new contextualized components using LLM

    Example:
        Invalid Components:
            - Knowledge of relational databases
            - SQL

        Fixed Components:
            - Knowledge of relational databases
            - Knowledge of relational SQL

Reason:
- Decomposing using LLM still has probability of hallucinating, we need to handle a invalid components
- Rule-based invalid components catcher are used to ensure the reliability of the system
- Invalid components and original Job Requirement are used for LLM to fix the invalid components

Limitation:
- Rule-based classification of components has limited flexibility, making it difficult to capture diverse and nuanced requirement structures.

Notes:
- Continuously analyze decomposed outputs to refine and expand the rule-based classification logic.
- If possible use ML's classification method to catch bad component

### f. Custom Retrieval or evidence preparation
Decision:
- Combined component-level retrieval with global retrieval using the original job requirement.
- Configurable Top-K for both component and global retrieval.

Reason:
- Component-level retrieval provides high precision but limited recall.
- Global retrieval (original job requirement) improves recall, enabling detection of indirect or implicit evidence.
- Avoided deduplication to preserve evidence diversity, ensuring each component retains sufficient supporting context.

Notes:
- Balanced precision–recall trade-off by separating local (component) and global (requirement-level) retrieval, improving robustness in noisy semantic matching scenarios.

### g. Job Requirement vs Evidence LLM
Decision:
- Performed LLM-based evaluation per individual job requirement (including its decomposed components)

Reason:
- Isolating evaluation at the requirement level reduces context noise and improves scoring accuracy from LLM.
- Enables more precise evidence-to-requirement alignment and clearer attribution of results.

Observed Issue:
- Avoided including pre-scored values in the prompt, as they can bias the LLM and lead it to treat those scores as evidence during evaluation. (e.g added Retrieval Score)
- Increasing the number of components beyond 3 per job requirement reduces the LLM’s ability to clearly separate components and their corresponding evidence, leading to degraded focus and evaluation quality.

### g. CV fit evaluation logic
Decision:
- Evaluation score is composed of:
  - Capability Level
  - Evidence Score
  - Responsibility Multiplier

Reason:
- Structured scoring constrains LLM reasoning into defined dimensions, reducing uncontrolled inference.
- Separating evaluation into three independent factors (capability, evidence strength, responsibility) prevents the LLM from collapsing all judgment into a single ambiguous score.
- Capability Level captures semantic alignment (does the evidence match the requirement).
- Evidence Score captures strength and context (how strong and reliable the evidence is).
- Responsibility Multiplier captures level of ownership or involvement, preventing overestimation from passive participation.
- Final score is computed outside the LLM to enforce deterministic aggregation and avoid arithmetic hallucination.
- Forcing structured reasoning output (capability + evidence + responsibility + reason) reduces free-form hallucination and improves consistency.

Limitation:
- The LLM may still misclassify one of the scoring dimensions (e.g., overestimate capability or misinterpret evidence strength).
- Errors in one dimension can propagate into the final score despite structured separation.
- Multiplicative scoring reduces the impact of overestimation in a single dimension but does not eliminate systematic bias.

Prompt:
"2. Determine capability level:
  - explicit_strong:
    - The capability is directly and clearly demonstrated
    - Evidence shows real implementation or usage
  - explicit_weak:
    - The capability is directly mentioned but shallow or weak
    - Evidence lacks depth, scale, or clear impact
  - implicit_strong:
    - Capability is not directly stated but strongly implied
    - Multiple signals support the inference
  - implicit_weak:
    - Capability is only loosely related
    - Weak or indirect signals
  - missing:
    - No meaningful evidence supports the component
    - Evidence is irrelevant or absent
3. Determine Evidence score:
  - Do NOT assume capabilities not stated in the evidence
  - Presence alone does NOT imply strength
  - If evidence type is not explicitly stated then default to the LOWER category.
  Evidence Score Mapping:
    - 1.0 → deployed system / real users / production impact
    - 0.8 → professional experience (internship, job, real company work)
    - 0.65 → solid project with clear implementation (non-trivial, end-to-end)
    - 0.5 → basic project
    - 0.4 → coursework / structured learning
    - 0.3 → listed skill, no proof
    - 0.2 → weak indirect relevance with some concrete supporting action
    - 0.0 → none
4. Determine Responsible Multiplier Score:
  - If ownership or leadership is not explicitly stated,
  - DO NOT assign "ownership" or "led", default to 1.0 .
  Responsible Multiplier:
    - 1.2 → led (led or managed others)
    - 1.1 → ownership (explicit OR clearly implied independent work) 
    - 1.0 → contributed (worked on or developed without ownership signals) 
    - 0.9 → assisted (helped, supported, or assisted)"
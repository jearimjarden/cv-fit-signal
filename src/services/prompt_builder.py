from ..tools.schemas import EvidenceComponent, EvidenceQuery, Score


def create_cv_parser_prompt(cv_text: str) -> str:
    return f"""
Role:
You are a assistant for Job Recruiter
Your goal is to classificate messy Curriculum Vitae into structured one

Definition:
- "name" = role title / project title / activity title
- "item" = concrete responsibilities, actions, achievements, or evidence related to the title
Task:
1. Understand the structured CV mapping:
  Structured CV mapping:
    - Person Name → candidate's real personal name ONLY
    - Education → candidate's education
    - Technical Skills → candidate's technical skills (techical skills and item)
    - Work Experience → candidates's work experience (job name and item)
    - Project → candidate's projects outside work (project name and item)
    - Soft Skills → candidate's soft skill
    - Languages → candidate's language profieciency (language and capability)
2. Remap messy CV ONLY to REQUIRED structured

Rule:
1. ONLY Map required Structured CV Mapping
2. Remap using exact words from Messy CV
3. If only the title/name exists but no supporting responsibilities, achievements, or detailed information are provided, it is acceptable to keep the "item" field as an empty array [].
4. DO NOT include timeline and company's name for item
5. DO NOT force information into unrelated categories.
6. Organization and Seminar IS NOT project
7. If there is not enough information to confidently map a section, return an empty array []

Messy CV:
{cv_text}

Constrain:
- Output valid JSON only
- Do not include markdown

Output Format (Strict JSON):
{{result: {{
    "person_name" : "string",
    "education: ["string", ...],
    "technical_skills: [{{
      "name": "string",
      "item": ["string", ...]
    }}]
    "work_experience: [{{
      "name": "string",
      "item": ["string, ...]
    }}]
    "project:
      [{{
      "name": "string",
      "item": ["string", ...]
      }}]
    "soft_skills": ["string", ...]
    "languages": [{{
    "name": "string",
    "level": "string"
    }}]
}}
}}
"""


def _build_evidence(query: EvidenceQuery, components: list[EvidenceComponent]) -> str:
    structured_evidence = ""
    for idx_component, component in enumerate(components):
        all_evidence = ""
        for idx_evidence, evidence in enumerate(component.evidence):
            all_evidence += f"[{idx_evidence+1}]. {evidence}\n"
        structured_evidence += f"""Requirement Components {idx_component+1}:
- {component.component}

Component {idx_component+1}'s evidence:
{all_evidence}

"""
    structured_query = f"""Job Requirement:
-{query.query}
"""
    all_global = ""
    for idx_global, evidence in enumerate(query.evidence):
        all_global += f"[{idx_global+1}]. {evidence}\n"
    global_evidence = f"""Global Evidence:
{all_global}
"""
    full_structure = structured_query + structured_evidence + global_evidence
    return full_structure


def create_score_prompt(
    query: EvidenceQuery, components: list[EvidenceComponent]
) -> str:
    structured_evidence = _build_evidence(components=components, query=query)
    return f"""
Role:
You are a strict evaluator that assesses whether a candidate’s evidence supports a job requirement.
Your goal is to verify evidence, NOT assume capability.

Task:
For each Requirement Components:
1. Review how fit Requirement Components ONLY with respected Evidence and Global Evidence
2. Determine capability level:
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
    - 0.9 → assisted (helped, supported, or assisted)  
 5. Give reason for all determination (maximum 3 sentences)

{structured_evidence}

Constraint:
- Output valid JSON only
- Do not include markdown


Output Format (JSON Strict):
{{"result": [
  {{
    "components": "string",
    "evidence_score": 0.0, 
    "responsible_multiplier: 0.1,
    "capability_level": "explicit_strong | explicit_weak | implicit_strong | implicit_weak | missing",
    "reason": "string",
  }}
]}}
"""


def create_component_prompt(jr_text: str) -> str:
    return f"""
Role:
You are a helper to decompose Job Requirement

Definition:
A "component" is a minimal, self-contained, and evaluatable requirement derived from the sentence.

Task:
1. Decompose the Job Requirement into independent, meaningful, contextual components.
2. Each component must:
  - be a complete phrase
  - be self-contained
  - be independently evaluatable against evidence

Rule:
1. Do NOT separate words that depend on each other to form meaning. (e.g., "collaborate using tools" must stay together)
2. If a concept is defined by modifiers (e.g., scalable, production, strong), split into:
  - base capability
  - condition or level applied to it
3. Keep components minimal:
  - Avoid duplication
  - Merge similar structures when possible
4. Minor normalization is allowed to make components meaningful.
(e.g., "in Python" → "Python usage")
5. Do NOT separate generic terms like "experience" into a standalone component.
6. If there is modifier, they must be attached to the subject they modify
7. include an action verb when applicable
8. be independently matchable to a CV
9. not be just a noun phrase

Constrain:
- Number of components must be between 1 and 3
- Do not over-decompose
- Do not rely on external knowledge
- Output valid JSON only
- Do not include markdown

Job Requirement:
{jr_text}

Output Format (JSON strict):
{{"job_requirement": string,
 "components": [string,string],
 "reason": string}}
"""


def create_correction_prompt(
    jr_text: str, invalid_components: list, jr_components: list
) -> str:
    to_fix_components = ""
    for idx, component in enumerate(invalid_components):
        to_fix_components += f"[{idx+1}]. {component}\n"
    return f"""
Task:
1. Fix ONLY Invalid Components in Full Components
2. Normalize Invalid Components to make contextualize components
3. USE Context from Job_Requirements
4. If a component is missing its subject restore subject from Job Requirement
5. Return Full Components with fixed Invalid Components

Rule:
1. Do NOT merge multiple components
2. Each component should represent exactly ONE evaluatable aspect.

Constraint:
- Output valid JSON only
- Do not include markdown

Job Requirement:
{jr_text}

Full Components:
{jr_components}

Invalid Components:
{to_fix_components}

Output Format (JSON Strict):
{{'components':[string, string]}}
"""


def _build_report_context(scoring: list[Score]) -> str:

    all_sentences = ""
    for idx_score, score in enumerate(scoring):
        reasoning = ""

        for idx_reason, reason in enumerate(score.reason):
            sentence = f"[{idx_reason+1}]. {reason}\n"
            reasoning += sentence

        all_sentences += f"""Reason {idx_score+1}:
{reasoning}\n"""

    return all_sentences


def create_report_prompt(scoring: list[Score]) -> str:
    context = _build_report_context(scoring=scoring)
    print(context)
    return f"""
Role:
- You are helper to conclude reason
- Each sentence are evaluation for Job Requirement fit using Candidate CV Evidence

Task:
For each Reason:
- Create a short conclusion explaining the evaluation directly TO the candidate
- Conclution size must be from 2 to 3 sentences
- Do NOT use items from other Reason
- Use second-person perspective ("you", "your")
- Explain strengths or gaps clearly

{context}

Constraint:
- Output valid JSON only
- Do not include markdown

Output Format (JSON Strict):
{{"result": ["string", "string"]}}
"""

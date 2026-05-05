from ...tools.schemas import EvidenceComponent, EvidenceQuery


def _build_evidence(query: EvidenceQuery, components: list[EvidenceComponent]):
    structured_evidence = """"""
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


def create_score_prompt(query: EvidenceQuery, components: list[EvidenceComponent]):
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
):
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

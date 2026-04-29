from networkx import constraint

roles = {1: """you are a recruiter assistant"""}

instruction = {1: """
For each requirement:
- Analyze the provided evidence
- Determine how well the evidence supports the requirement
- Give a score from 0 to 1
- Provide a short reasoning with and cite the evidence (include the evidence number)
               """}

output_format = {1: """
Schema:
{
  "results": [
    {
      "requirement: str,
      "match": "strong" | "medium" | "weak",
      "score": float,
      "reason": string
    }
  ]
}"""}

constraint = {1: """
Rules:
- Do not add any text outside JSON
- Do not explain the format
- Do not include markdown
- Output valid JSON only"""}


def build_context(jr_chunks: list, cv_chunks: list):
    all_context = ""
    for idx_jr, jr in enumerate(jr_chunks):
        evidences = "\n".join(
            f"[{idx_chunks+1}] {evidence}"
            for idx_chunks, evidence in enumerate(cv_chunks[idx_jr])
        )
        context = f"""
== Requirement {idx_jr+1} ==
{jr}

-- Evidence --
{evidences}
"""
        all_context += context

    return all_context


def build_prompt(jr_chunks: list[list[str]], cv_chunks: list[list[str]]) -> str:
    context = build_context(jr_chunks=jr_chunks, cv_chunks=cv_chunks)
    return f"""
{roles[1]}

CONTEXT:
{context}

INSTRUCTION:
{instruction[1]}

OUTPUT FORMAT:
{output_format[1]}

CONSTRAINT:
{constraint[1]}
"""

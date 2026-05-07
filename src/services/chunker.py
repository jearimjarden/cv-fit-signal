from ..tools.schemas import CVChunk, StructuredCVItem, StructuredCVLanguage
from ..tools.schemas import JRChunks
from .prompt_builder import create_component_prompt, create_correction_prompt
from .llm_client import LLMClient
import json


def decompose_and_validate_jr(
    jr_parsed_text: list, llm_client: LLMClient
) -> list[JRChunks]:
    all_chunks = []

    for idx, jr_text in enumerate(jr_parsed_text):
        prompt = create_component_prompt(jr_text=jr_text)
        response = llm_client.generate(prompt=prompt)
        dict_response = json.loads(response)
        dict_response["idx"] = idx
        all_chunks.append(JRChunks(**dict_response))

    validated_chunks = _validate_jr_chunks(jr_chunks=all_chunks, llm_client=llm_client)

    return validated_chunks


def _validate_jr_chunks(
    jr_chunks: list[JRChunks], llm_client: LLMClient
) -> list[JRChunks]:
    validated_jr_components = []
    for jr_decom in jr_chunks:
        invalid_components = []

        for component in jr_decom.components:
            c = component.lower().strip()
            if len(c.split()) < 2 or c.startswith(
                ("for ", "in ", "with ", "using ", "to ")
            ):
                invalid_components.append(c)

        if invalid_components:
            prompt = create_correction_prompt(
                jr_text=jr_decom.job_requirement,
                invalid_components=invalid_components,
                jr_components=jr_decom.components,
            )
            response = llm_client.generate(prompt=prompt)
            dict_answer = json.loads(response)
            validated_jr_components.append(
                JRChunks(
                    idx=jr_decom.idx,
                    job_requirement=jr_decom.job_requirement,
                    components=dict_answer["components"],
                    reason=jr_decom.reason,
                )
            )
        else:
            validated_jr_components.append(jr_decom)

    return validated_jr_components


def chunk_cv_semantic(
    technical_skills: list[StructuredCVItem],
    work_experiences: list[StructuredCVItem],
    projects: list[StructuredCVItem],
    languages: list[StructuredCVLanguage],
    soft_skills: list[str],
) -> list[CVChunk]:
    chunk_idx = 0
    all_chunk = []

    for skill in technical_skills:
        skills = ", ".join(skill.item)
        if skills:
            chunk = f"Technical Skills ({skill.name}): {skills}"
        else:
            chunk = f"Technical Skills: {skill.name}"
        all_chunk.append(CVChunk(idx=chunk_idx, type="Technical Skill", chunk=chunk))
        chunk_idx += 1

    for experience in work_experiences:
        for item in experience.item:
            if item:
                chunk = f"Work Experience ({experience.name}): {item}"
            else:
                chunk = f"Work Experience: {experience.name}"

            all_chunk.append(
                CVChunk(idx=chunk_idx, type="Work Experience", chunk=chunk)
            )
            chunk_idx += 1

    for project in projects:
        for item in project.item:
            if item:
                chunk = f"Project ({project.name}): {item}"
            else:
                chunk = f"Project: {project.name}"

            all_chunk.append(CVChunk(idx=chunk_idx, type="Project", chunk=chunk))
            chunk_idx += 1

    for language in languages:
        if language.level:
            chunk = f"Language: {language.name} ({language.level})"
        else:
            chunk = f"Language: {language.name}"

        all_chunk.append(CVChunk(idx=chunk_idx, type="Language", chunk=chunk))
        chunk_idx += 1

    for idx in range(0, len(soft_skills), 2):
        soft_skill = ", ".join(soft_skills[idx : idx + 3])
        chunk = f"Soft Skills: {soft_skill}"
        all_chunk.append(CVChunk(idx=chunk_idx, type="Soft Skills", chunk=chunk))
        chunk_idx += 1

    return all_chunk


def _legacy_chunk_cv(text_experience: str, text_skills: str) -> list:
    """Legacy custom CV chunking:
    - Added title/category name for each chunk
    - Added subtile name (project name) for experience

    Notes: Unused for active pipeline"""

    experience_chunks = []
    print(text_experience)
    experiences_splitted = text_experience.split("\n")
    experiences_normalized = [x for x in experiences_splitted[:] if x]

    context = ""
    subtitle = ""
    for experience in experiences_normalized:
        if experience.strip().startswith("-"):
            context = experience
        elif not experience.strip().startswith("-"):
            subtitle = experience
        if subtitle and context:
            experience_chunks.append(f"Experience: {subtitle} {context}")
            context = ""

    skills_chunks = []
    skills_splitted = text_skills.split("\n")
    skills_normalized = [x for x in skills_splitted if x]

    for skill in skills_normalized:
        skill = skill.replace("-", "")
        skill.strip()
        skills_chunks.append(f"Skills:{skill}")

    return experience_chunks + skills_chunks

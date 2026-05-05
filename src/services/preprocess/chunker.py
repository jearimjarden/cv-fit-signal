def semantic_chunk(text_experience: str, text_skills: str) -> list:
    experience_chunks = []
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

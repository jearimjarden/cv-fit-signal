from openai import OpenAI


def call_llm_oa(prompt: str, oa_api_key: str) -> str:
    client = OpenAI(api_key=oa_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # important for RAG (more deterministic)
    )

    if isinstance(response.choices[0].message.content, str):
        return response.choices[0].message.content

    else:
        raise Exception

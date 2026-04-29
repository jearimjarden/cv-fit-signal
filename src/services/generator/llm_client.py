from openai import OpenAI


def call_llm_oa(prompt: str, oa_api_key: str):
    client = OpenAI(api_key=oa_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # important for RAG (more deterministic)
    )

    return response.choices[0].message.content

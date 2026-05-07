from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content

        if isinstance(content, str):
            return content

        else:
            raise Exception("LLM Output isn't string")

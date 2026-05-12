from openai import AuthenticationError, OpenAI, APITimeoutError, APIConnectionError
import json
import logging
from ..services.prompt_builder import create_fix_json_prompt
from ..tools.exceptions_schemas import (
    InvalidJSON,
    InvalidResponse,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
)
from ..tools.schemas import Config, InferenceStage, PipelineStage, PreprocessStage
from ..tools.observabillity import TrackToken

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        track_token: TrackToken,
        config: Config,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.track_token = track_token
        self.config = config

    def generate(self, prompt: str, stage: InferenceStage | PreprocessStage) -> str:
        for attempt in range(self.config.llm.max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=self.config.llm.timeout,
                )
                if response.usage:
                    self.track_token.add(
                        completion_tokens=response.usage.completion_tokens,
                        prompt_token=response.usage.prompt_tokens,
                        stage=stage.value,
                    )

                content = response.choices[0].message.content

                if isinstance(content, str):
                    return content

                else:
                    raise InvalidResponse("LLM Output isn't string")

            except AuthenticationError:
                raise LLMAuthenticationError("Invalid OpenAI API key")

            except APITimeoutError:
                logger.warning(
                    f"LLM timeout on retry, {attempt + 1}/{self.config.llm.max_retry}",
                    extra={"stage": PipelineStage.LLM},
                )

                if attempt == self.config.llm.max_retry - 1:
                    raise LLMTimeoutError(
                        f"LLM request timed out after {self.config.llm.max_retry} attempts"
                    )

            except APIConnectionError:
                raise LLMConnectionError("Failed to connect to OpenAI API")

        raise LLMError(
            "Unknown error occured in LLM generation", stage=PipelineStage.LLM
        )

    def json_repair(self, context: str) -> dict:
        prompt = create_fix_json_prompt(context=context)

        for attempt in range(self.config.llm.max_retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=self.config.llm.timeout,
                )

                if response.usage:
                    self.track_token.add(
                        completion_tokens=response.usage.completion_tokens,
                        prompt_token=response.usage.prompt_tokens,
                        stage=PipelineStage.LLMRepair,
                    )

                content = response.choices[0].message.content

                if isinstance(content, str):
                    dict_content = json.loads(content)
                    return dict_content
                else:
                    raise InvalidResponse("LLM response content is not a string")

            except json.JSONDecodeError:
                raise InvalidJSON("Failed to repair invalid JSON response")

            except APITimeoutError:
                logger.warning(
                    f"LLM timeout on retry, {attempt + 1}/{self.config.llm.max_retry}",
                    extra={"stage": PipelineStage.LLM},
                )

                if attempt == self.config.llm.max_retry - 1:
                    raise LLMTimeoutError(
                        f"LLM request timed out after {self.config.llm.max_retry} attempts"
                    )

            except APIConnectionError:
                raise LLMConnectionError("Failed to connect to OpenAI API")

        raise LLMError(
            "Unknown error occured in LLM generation", stage=PipelineStage.LLM
        )

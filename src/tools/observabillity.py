import logging
import time
from functools import wraps
from .schemas import ConfigLLM, LatencyStored, TokenSummary

logger = logging.getLogger(__name__)


class LatencyStore:
    def __init__(self):
        self.latencies = {}

    def add(self, stage: str, latency_ms: float):
        self.latencies[stage] = latency_ms

    def get_all(self) -> LatencyStored:
        return LatencyStored(latencies_ms=self.latencies)


def track_latency(stage: str):
    def decorator(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)
                latency_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )

                self.latency_store.add(
                    stage=stage,
                    latency_ms=latency_ms,
                )

                return result

            except Exception:
                latency_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )
                self.latency_store.add(
                    stage=f"{stage}_failed",
                    latency_ms=latency_ms,
                )
                raise

        return wrapper

    return decorator


class TrackToken:
    def __init__(self, llm_config: ConfigLLM):
        self.tokens_history = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_idr = 0.0
        self.llm_config = llm_config

    def add(self, completion_tokens: int, prompt_token: int, stage: str):
        if stage not in self.tokens_history:
            self.tokens_history[stage] = {
                "prompt_token": 0,
                "completion_tokens": 0,
            }

        self.tokens_history[stage]["completion_tokens"] += completion_tokens
        self.tokens_history[stage]["prompt_token"] += prompt_token
        self.total_prompt_tokens += prompt_token
        self.total_completion_tokens += completion_tokens
        self.total_cost_idr += (
            (
                (self.llm_config.prompt_tokens_per_1M * prompt_token)
                + (self.llm_config.completions_tokens_per_1M * completion_tokens)
            )
            / 1000000
            * self.llm_config.usd_to_idr
        )

    def get_all(self) -> TokenSummary:
        return TokenSummary(
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_cost_idr=self.total_cost_idr,
            tokens_history=self.tokens_history,
        )

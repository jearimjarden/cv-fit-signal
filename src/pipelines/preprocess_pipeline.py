from typing import TypeVar, Type
from pydantic import ValidationError
import logging
from ..services.embedder import EmbeddingService
from ..services.chunker import chunk_cv_semantic
from ..services.llm_client import LLMClient
from ..services.parser import parse_cv_llm
from ..tools.exceptions_schemas import (
    PreprocessorError,
    InvalidCVError,
    LLMError,
    LoggedError,
)
from ..tools.observabillity import LatencyStore, TrackToken
from ..tools.schemas import (
    CVChunk,
    CVEmbedding,
    CVInput,
    CVSelection,
    Env,
    Config,
    LatencyStored,
    PreprocessStage,
    PipelineStage,
    StructuredCV,
    TokenSummary,
)
from ..tools.observabillity import track_latency
from ..IO.save_cv import save_all_cv

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    def __init__(
        self,
        config: Config,
        settings: Env,
        track_token: TrackToken,
        latency_store: LatencyStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.settings = settings
        self.config = config
        self.track_token = track_token
        self.latency_store = latency_store
        self.llm_client = LLMClient(
            api_key=self.settings.oa_api_key,
            track_token=self.track_token,
            config=self.config,
            model=self.config.llm.model,
        )
        self.embedding_service = embedding_service

    def run(self, cv_input: CVInput, cv_name: CVSelection) -> None:
        try:
            self.preprocess_cv(cv_input=cv_input, cv_name=cv_name)

            logger.info(
                "CV Created",
                extra={
                    "latencies": self.latency_store.get_all().latencies_ms,
                },
            )

            logger.info(
                "Token Tracked",
                extra={"summary": self.track_token.get_all()},
            )

        except (PreprocessorError, LLMError) as e:
            logger.error(str(e), extra={"stage": e.stage})
            raise LoggedError from e

        except ValidationError as e:
            logger.error(str(e), extra={"stage": PipelineStage.PREPROCESS})
            raise LoggedError from e

    @track_latency(stage=PipelineStage.PREPROCESS)
    def preprocess_cv(self, cv_input: CVInput, cv_name: CVSelection) -> None:
        logger.info(
            "CV input accepted",
            extra={"stage": PreprocessStage.INPUT, "cv_text_n": len(cv_input.text)},
        )

        cv_parsed = self.parse_cv(cv_input=cv_input)
        logger.info(
            "Parsed CV",
            extra={"stage": PreprocessStage.PARSE, "cv_parsed_n": len(cv_parsed)},
        )

        cv_chunks = self.chunk_cv(cv_parsed=cv_parsed)
        logger.info(
            "Chunked parsed CV",
            extra={
                "stage": PreprocessStage.CHUNK,
                "cv_chunks(n)": len(cv_chunks),
            },
        )

        cv_embedding = self.embed_cv(cv_chunks=cv_chunks)
        logger.info(
            "Embedded CV chunks",
            extra={
                "stage": PreprocessStage.EMBED,
            },
        )

        self.save_cv(
            cv_parsed=cv_parsed,
            cv_chunk=cv_chunks,
            cv_embedding=cv_embedding,
            cv_name=cv_name,
            token_summary=self.track_token.get_all(),
            latency_stored=self.latency_store.get_all(),
        )

    @track_latency(stage=PreprocessStage.PARSE)
    def parse_cv(self, cv_input: CVInput) -> StructuredCV:
        if len(cv_input.text) < 500:
            raise InvalidCVError(
                f"Not Enough CV Information, cv_text_n: {len(cv_input.text)}",
            )

        structured_cv = parse_cv_llm(cv_text=cv_input.text, llm_client=self.llm_client)

        if len(structured_cv) < 10:
            raise InvalidCVError(
                f"Not Enough CV Information, cv_parse_n: {len(structured_cv)}",
            )

        return structured_cv

    def chunk_cv(self, cv_parsed: StructuredCV) -> list[CVChunk]:
        return chunk_cv_semantic(
            technical_skills=cv_parsed.technical_skills,
            work_experiences=cv_parsed.work_experience,
            projects=cv_parsed.project,
            languages=cv_parsed.languages,
            soft_skills=cv_parsed.soft_skills,
        )

    def embed_cv(self, cv_chunks: list[CVChunk]) -> list[CVEmbedding]:
        return self.embedding_service.embed_cv(
            cv_chunks=cv_chunks, batch_size=self.config.embedding.batch_size
        )

    def save_cv(
        self,
        cv_parsed: StructuredCV,
        cv_chunk: list[CVChunk],
        cv_embedding: list[CVEmbedding],
        cv_name: CVSelection,
        token_summary: TokenSummary,
        latency_stored: LatencyStored,
    ) -> None:
        cv_name_str = cv_name.text.strip().lower()

        save_all_cv(
            cv_parsed=cv_parsed,
            cv_chunk=cv_chunk,
            cv_embedding=cv_embedding,
            cv_name_str=cv_name_str,
            token_summary=token_summary,
            latency_stored=latency_stored,
        )

        logger.info(
            f"CV artifact created for '{cv_name_str}'",
            extra={"stage": PreprocessStage.SAVE},
        )

    T = TypeVar("T", bound="PreprocessPipeline")

    @classmethod
    def start_from_config(
        cls: Type[T],
        config: Config,
        settings: Env,
        track_token: TrackToken,
        latency_store: LatencyStore,
        embedding_service: EmbeddingService,
    ) -> T:
        return cls(
            config=config,
            settings=settings,
            track_token=track_token,
            latency_store=latency_store,
            embedding_service=embedding_service,
        )

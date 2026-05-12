from typing import Type, TypeVar
from pydantic import ValidationError
from .preprocess_pipeline import PreprocessPipeline
from ..tools.exceptions_schemas import (
    InferenceError,
    PreprocessorError,
    InvalidFileError,
    LLMError,
    LoggedError,
)
from ..IO.load_cv import load_cv_all
from ..services.evaluator import EvaluatorService
from ..services.llm_client import LLMClient
from ..services.parser import parse_normalize_jr
from ..services.embedder import EmbeddingService
from ..services.retriever import (
    faiss_ip_search,
    retrieve_base_chunk,
)
from ..services.chunker import decompose_and_validate_jr
from ..IO.json_loader import load_json
from ..tools.observabillity import LatencyStore, TrackToken, track_latency
from ..tools.schemas import (
    BaseRetrieval,
    CVChunk,
    CVEmbedding,
    CVInput,
    Config,
    Env,
    InferenceStage,
    InputMode,
    JRChunks,
    JREmbedding,
    JRInput,
    Report,
    PipelineStage,
)
import logging

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(
        self,
        config: Config,
        settings: Env,
        latency_store: LatencyStore,
        track_token: TrackToken,
    ) -> None:
        self.settings = settings
        self.config = config
        self.latency_store = latency_store
        self.track_token = track_token
        self.llm_client = LLMClient(
            api_key=self.settings.oa_api_key,
            track_token=self.track_token,
            config=self.config,
            model=self.config.llm.model,
        )
        self.embedding_service = EmbeddingService(
            device=self.config.embedding.device, latency_store=self.latency_store
        )
        self.evaluator_service = EvaluatorService(
            llm_client=self.llm_client,
            evaluation=self.config.evaluation,
            latency_store=self.latency_store,
        )
        self.preprocess = PreprocessPipeline(
            config=self.config,
            settings=self.settings,
            track_token=self.track_token,
            latency_store=self.latency_store,
            embedding_service=self.embedding_service,
        )

    def run(self, cv_selection: str, jr_input: JRInput) -> Report:
        try:
            if self.config.input_mode == InputMode.FILE:
                report = self.predict_file()
                logger.info(
                    "File predicted",
                    extra={
                        "stage": PipelineStage.INFERENCE,
                        "latencies": self.latency_store.get_all().latencies_ms,
                    },
                )

                logger.info(
                    "Token Tracked",
                    extra={
                        "stage": PipelineStage.INFERENCE,
                        "summary": self.track_token.get_all(),
                    },
                )

                return report

            elif self.config.input_mode == InputMode.API:
                report = self.predict_api(cv_selection=cv_selection, jr_input=jr_input)
                logger.info(
                    "API predicted",
                    extra={
                        "stage": PipelineStage.INFERENCE,
                        "latencies": self.latency_store.get_all().latencies_ms,
                    },
                )

                logger.info(
                    "Token Tracked",
                    extra={
                        "stage": PipelineStage.INFERENCE,
                        "summary": self.track_token.get_all(),
                    },
                )

                return report

            raise LLMError(
                "Inference Pipeline Failed Unexpectedly", stage=PipelineStage.INFERENCE
            )

        except (LLMError, InferenceError, PreprocessorError) as e:
            logger.error(str(e), extra={"stage": e.stage})
            raise LoggedError from e

        except ValidationError as e:
            messages = []

            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])

                if err["type"] == "missing":
                    messages.append(f"Missing config parameter: '{field}'")

                elif err["type"] == "extra_forbidden":
                    messages.append(f"Forbidden extra config parameter: '{field}'")

                else:
                    messages.append(f"Invalid config value for '{field}': {err['msg']}")

            logger.error(" | ".join(messages), extra={"stage": PipelineStage.INFERENCE})

            raise LoggedError from e

    @track_latency(stage=InferenceStage.PREDICTFILE)
    def predict_file(self) -> Report:

        cv_input, jr_input = self.load_cv_jr_file()
        logger.info(
            "Retrieved raw CV and JR input",
            extra={
                "stage": InferenceStage.FILEINPUT,
                "cv_length(n)": len(cv_input.text),
                "jr_length(n)": len(jr_input.text),
            },
        )

        cv_parsed = self.preprocess.parse_cv(cv_input=cv_input)
        jr_parsed = self.parse_jr(jr_input=jr_input)
        logger.info(
            "Parsed raw CV and JR",
            extra={
                "stage": InferenceStage.PARSE,
                "cv_parsed(n)": len(cv_parsed),
                "jr_parsed(n)": len(jr_parsed),
            },
        )

        cv_chunks = self.preprocess.chunk_cv(cv_parsed=cv_parsed)
        jr_chunks = self.chunk_jr(jr_parsed_text=jr_parsed)
        logger.info(
            "Chunked parsed CV and JR",
            extra={
                "stage": InferenceStage.CHUNK,
                "cv_chunks(n)": len(cv_chunks),
                "jr_chunks(n)": sum(len(chunk) for chunk in jr_chunks),
            },
        )

        jr_embedding = self.embedding_service.embed_jr(
            jr_chunks=jr_chunks, batch_size=self.config.embedding.batch_size
        )
        logger.info(
            "Embedded JR chunk",
            extra={
                "stage": InferenceStage.EMBED,
            },
        )

        cv_embedding = self.preprocess.embed_cv(cv_chunks=cv_chunks)
        logger.info(
            "Embedded CV chunk",
            extra={
                "stage": InferenceStage.EMBED,
            },
        )

        retrieved_base = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )
        logger.info(
            "Retrieved similar semantic chunk",
            extra={
                "stage": InferenceStage.RETRIEVAL,
                "base_retrieval(n)": sum(
                    len(retrieved) for retrieved in retrieved_base
                ),
            },
        )

        evaluations = self.evaluator_service.generate_evaluation(
            base_retrieval=retrieved_base
        )
        logger.info(
            "Successfully generated evaluation",
            extra={
                "stage": InferenceStage.EVALUATION,
                "evaluations(n)": len(evaluations),
            },
        )

        scores = self.evaluator_service.generate_score(evaluations=evaluations)
        logger.info(
            "Successfully generated score",
            extra={
                "stage": InferenceStage.SCORING,
                "scores": [score.score for score in scores],
            },
        )

        reports = self.evaluator_service.generate_report(
            scores=scores, candidate_name=cv_parsed.person_name
        )

        logger.info(
            "Successfully generated report",
            extra={"stage": InferenceStage.REPORT, "final_score": reports.report_score},
        )

        return reports

    @track_latency(stage=InferenceStage.PREDICTAPI)
    def predict_api(self, cv_selection: str, jr_input: JRInput) -> Report:
        metadata, cv_chunks, cv_embedding = self.load_cv(cv_selection=cv_selection)
        logger.info(
            "CV acquired",
            extra={
                "stage": InferenceStage.APIINPUT,
                "person_name": metadata.cv_name,
                "cv_chunks_n": metadata.cv_chunked_n,
            },
        )
        jr_parsed = self.parse_jr(jr_input=jr_input)
        logger.info(
            "Parsed raw JR",
            extra={
                "stage": InferenceStage.PARSE,
                "jr_parsed(n)": len(jr_parsed),
            },
        )

        jr_chunks = self.chunk_jr(jr_parsed_text=jr_parsed)
        logger.info(
            "Chunked parsed JR",
            extra={
                "stage": InferenceStage.CHUNK,
                "jr_chunks(n)": sum(len(chunk) for chunk in jr_chunks),
            },
        )
        jr_embedding = self.embedding_service.embed_jr(
            jr_chunks=jr_chunks, batch_size=self.config.embedding.batch_size
        )
        logger.info(
            "Embedded chunked JR",
            extra={
                "stage": InferenceStage.EMBED,
            },
        )

        retrieved_base = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )
        logger.info(
            "Retrieved similar semantic chunk",
            extra={
                "stage": InferenceStage.RETRIEVAL,
                "base_retrieval(n)": sum(
                    len(retrieved) for retrieved in retrieved_base
                ),
            },
        )

        evaluations = self.evaluator_service.generate_evaluation(
            base_retrieval=retrieved_base
        )
        logger.info(
            "Successfully generated evaluation",
            extra={
                "stage": InferenceStage.EVALUATION,
                "evaluations(n)": len(evaluations),
            },
        )

        scores = self.evaluator_service.generate_score(evaluations=evaluations)
        logger.info(
            "Successfully generated score",
            extra={
                "stage": InferenceStage.SCORING,
                "scores": [score.score for score in scores],
            },
        )

        reports = self.evaluator_service.generate_report(
            scores=scores, candidate_name=metadata.cv_name
        )

        logger.info(
            "Successfully generated report",
            extra={"stage": InferenceStage.REPORT, "final_score": reports.report_score},
        )

        return reports

    def load_cv(self, cv_selection: str):
        return load_cv_all(cv_selection=cv_selection)

    @track_latency(stage=InferenceStage.CHUNK)
    def chunk_jr(
        self,
        jr_parsed_text: list[str],
    ) -> list[JRChunks]:
        return decompose_and_validate_jr(
            jr_parsed_text=jr_parsed_text, llm_client=self.llm_client
        )

    def parse_jr(self, jr_input: JRInput) -> list[str]:
        """Default Job Requirement Parser using New Line and Normalization"""

        parsed_normalized_jr = parse_normalize_jr(
            text=jr_input.text,
            chunk_size=self.config.jr_chunk.chunk_size,
            stride=self.config.jr_chunk.stride,
        )
        return parsed_normalized_jr

    def retrieve_base(
        self,
        cv_embedding: list[CVEmbedding],
        jr_embedding: list[JREmbedding],
        cv_chunks: list[CVChunk],
    ) -> list[BaseRetrieval]:
        search_result = faiss_ip_search(
            cv_embedding=cv_embedding,
            jr_embedding=jr_embedding,
            query_top_k=self.config.retrieval.query_top_k,
            component_top_k=self.config.retrieval.component_top_k,
        )

        idx_to_chunk = {item.idx: item.chunk for item in cv_chunks}

        retrieved_chunks = retrieve_base_chunk(
            search_result=search_result,
            idx_to_chunk=idx_to_chunk,
            threshold=self.config.retrieval.threshold,
            filter_below_threshold=self.config.retrieval.filter_below_threshold,
        )

        logger.debug(
            "Retrieved Chunk",
            extra={"stage": InferenceStage.RETRIEVAL, "result": retrieved_chunks},
        )
        return retrieved_chunks

    def load_cv_jr_file(self) -> tuple[CVInput, JRInput]:
        cv_input = self._load_CV()
        jr_input = self._load_JR()

        return cv_input, jr_input

    def _load_JR(self) -> JRInput:
        file_name = self.config.file_service.jr.file_name
        folder_path = self.config.file_service.jr.folder_path

        if file_name is None or folder_path is None:
            raise InvalidFileError(
                "file_name or folder_path can not be empty when using File Service"
            )

        jr_loaded = load_json(
            file_name=file_name,
            folder_path=folder_path,
        )

        validated_jr = JRInput(**jr_loaded)
        return validated_jr

    def _load_CV(self) -> CVInput:
        file_name = self.config.file_service.cv.file_name
        folder_path = self.config.file_service.cv.folder_path

        if file_name is None or folder_path is None:
            raise InvalidFileError(
                "file_name or folder_path can not be empty when using File Service"
            )

        cv_loaded = load_json(
            file_name=file_name,
            folder_path=folder_path,
        )

        validated_cv = CVInput(**cv_loaded)
        return validated_cv

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def load_from_config(
        cls: Type[T],
        config: Config,
        setting: Env,
        latency_store: LatencyStore,
        track_token: TrackToken,
    ) -> T:
        return cls(config, setting, latency_store, track_token)

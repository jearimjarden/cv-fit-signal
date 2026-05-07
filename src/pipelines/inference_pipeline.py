from typing import Type, TypeVar
from ..services.evaluator import EvaluatorService
from ..services.llm_client import LLMClient
from ..services.parser import (
    parse_cv_llm,
    parse_normalize_jr,
)
from ..services.embedder import EmbeddingService
from ..services.retriever import (
    faiss_ip_search,
    retrieve_base_chunk,
)
from ..services.chunker import chunk_cv_semantic, decompose_and_validate_jr
from ..IO.CV_loader_creator import load_CV
from ..IO.JR_loader_creator import load_JR
from ..tools.schemas import (
    BaseRetrieval,
    CVChunk,
    CVEmbedding,
    CVInput,
    Config,
    Env,
    JRChunks,
    JREmbedding,
    Report,
    StructuredCV,
)


class InferencePipeline:
    def __init__(self, config: Config, settings: Env) -> None:
        self.settings = settings
        self.config = config
        self.llm_client = LLMClient(api_key=self.settings.oa_api_key)
        self.embedding_service = EmbeddingService(device=self.config.embedding.device)
        self.evaluator_service = EvaluatorService(
            llm_client=self.llm_client, evaluation=self.config.evaluation
        )

    def run(self) -> Report:
        cv_input = self._load_CV()
        jr_text = self._load_JR()

        cv_parsed = self.parse_cv(cv_input=cv_input)
        jr_parsed = self.parse_jr(jr_text)

        cv_chunks = self.chunk_cv(cv_parsed=cv_parsed)
        print(cv_chunks)
        jr_chunks = self.chunk_jr(jr_parsed_text=jr_parsed)

        jr_embedding, cv_embedding = self.embedding_service.embed_chunks(
            cv_chunks=cv_chunks,
            batch_size=self.config.embedding.batch_size,
            jr_chunks=jr_chunks,
        )

        retrieved_base = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )

        evaluation = self.evaluator_service.generate_evaluation(
            base_retrieval=retrieved_base,
            query_rerank=self.config.retrieval.query_top_k,
            component_rerank=self.config.retrieval.component_top_k,
        )

        score = self.evaluator_service.generate_score(evaluations=evaluation)

        report = self.evaluator_service.generate_report(
            score=score, candidate_name=cv_parsed.person_name
        )

        print(report)
        # Show Report
        print(f"Date: {report.datetime}")
        print(f"Name: {report.name}")
        for r in report.report:
            print(f"Query: {r.query}")
            print(f"Score: {r.score}")
            print(f"Reason: {r.reason}")
        print(f"Final Score: {report.final_score}")

        return report

    def chunk_jr(
        self,
        jr_parsed_text: list[str],
    ) -> list[JRChunks]:
        return decompose_and_validate_jr(
            jr_parsed_text=jr_parsed_text, llm_client=self.llm_client
        )

    def chunk_cv(self, cv_parsed: StructuredCV) -> list[CVChunk]:
        return chunk_cv_semantic(
            technical_skills=cv_parsed.technical_skills,
            work_experiences=cv_parsed.work_experience,
            projects=cv_parsed.project,
            languages=cv_parsed.languages,
            soft_skills=cv_parsed.soft_skills,
        )

    def parse_cv(self, cv_input: CVInput):
        return parse_cv_llm(cv_text=cv_input.text, llm_client=self.llm_client)

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
            search_result=search_result, idx_to_chunk=idx_to_chunk
        )
        return retrieved_chunks

    def parse_jr(self, text: str) -> list[str]:
        """Default Job Requirement Parser using New Line and Normalization"""

        parsed_normalized_jr = parse_normalize_jr(
            text=text,
            chunk_size=self.config.jr_chunk.chunk_size,
            stride=self.config.jr_chunk.stride,
        )
        return parsed_normalized_jr

    def _load_JR(self) -> str:
        return load_JR(
            file_name=self.config.file_service.jr.file_name,
            folder_path=self.config.file_service.jr.folder_path,
        )

    def _load_CV(self) -> CVInput:
        cv = load_CV(
            file_name=self.config.file_service.cv.file_name,
            folder_path=self.config.file_service.cv.folder_path,
        )

        validated_cv = CVInput(**cv)

        return validated_cv

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def load_from_config(cls: Type[T], config: Config, setting: Env) -> T:
        return cls(config, setting)

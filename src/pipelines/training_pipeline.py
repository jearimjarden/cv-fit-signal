from typing import Type, TypeVar
import shutil
import numpy as np
from ..services.generator.llm_client import call_llm_oa
from ..services.augmentator.prompt_builder import (
    create_component_prompt,
    create_correction_prompt,
    create_score_prompt,
)
from ..IO.CV_loader_creator import load_CV
from ..IO.JR_loader_creator import load_JR
from ..IO.report_creator import save_report
from ..tools.schemas import (
    BaseRetrieval,
    Config,
    ChunkingMethod,
    EmbeddingMethod,
    Env,
    Evaluation,
    Evidence,
    JRDecomposed,
    JREmbed,
)
from ..services.preprocess.parser import (
    parse_text_regex,
    parse_text_nltk,
    parse_text_nl,
    parse_text_regex_nl,
    parse_text_structured,
)
from ..services.preprocess.embedder import embed_chunk_cpu, embed_chunk_cuda
from ..services.retrieval.vector_search import (
    faiss_ip_search,
    prepare_evidence,
    retrieve_base_chunk,
    retrieve_top_k,
)
from ..services.evaluator.evaluation import (
    create_evaluation_report,
    print_evaluation_report,
)
from ..services.preprocess.chunker import semantic_chunk
import json


class TrainingPipeline:
    def __init__(self, config: Config, settings: Env) -> None:
        self.settings = settings
        self.config = config

    def run(self):
        jr_text = self._load_JR()
        cv_text = self._load_CV()

        jr_parsed_text = self.parse_text_jr(jr_text)
        cv_parsed_text = self.parse_text_cv(cv_text)

        cv_chunks = self.chunk_cv(cv_parsed_text)
        jr_chunks = self.chunk_jr(jr_parsed_text=jr_parsed_text)
        print(jr_chunks)

        jr_embedding, cv_embedding = self.embed_chunks(
            cv_chunks=cv_chunks, jr_decomposed=jr_chunks
        )

        retrieved_chunks = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )

        prepared_evidence = self.prepare_retrieval_evidence(
            retrieved_chunks=retrieved_chunks
        )

        answers = self.generate_score(prepared_evidence=prepared_evidence)
        # for answer in answers:
        #     print(answer)

        # self.evaluate(
        #     cv_chunks=cv_chunks,
        #     jr_chunks=jr_raw_chunks,
        #     distances=distances,
        #     indices=indices,
        #     answers=answers,
        # )

    def prepare_retrieval_evidence(self, retrieved_chunks: list[BaseRetrieval]):
        reranked_retrieval = prepare_evidence(
            base_retrieval=retrieved_chunks,
            query_rerank=self.config.training.retrieval.query_rerank,
            component_rerank=self.config.training.retrieval.component_rerank,
        )
        return reranked_retrieval

    def _validate_jr_components(self, jr_decomposes: list[JRDecomposed]):
        validated_jr_components = []
        for jr_decom in jr_decomposes:
            invalid_components = []
            print("validating")
            for component in jr_decom.components:
                c = component.lower().strip()
                if len(c.split()) < 2 or c.startswith(
                    ("for ", "in ", "with ", "using ", "to ")
                ):
                    invalid_components.append(c)
                    print("invalid")

            if invalid_components:
                prompt = create_correction_prompt(
                    jr_text=jr_decom.job_requirement,
                    invalid_components=invalid_components,
                    jr_components=jr_decom.components,
                )
                answer = call_llm_oa(prompt=prompt, oa_api_key=self.settings.oa_api_key)
                dict_answer = json.loads(answer)
                validated_jr_components.append(
                    JRDecomposed(
                        idx=jr_decom.idx,
                        job_requirement=jr_decom.job_requirement,
                        components=dict_answer["components"],
                        reason=jr_decom.reason,
                    )
                )
            else:
                validated_jr_components.append(jr_decom)
        return validated_jr_components

    def _get_jr_components(self, jr_parsed_text: list) -> list[JRDecomposed]:
        all_components = []
        for idx, jr_text in enumerate(jr_parsed_text):
            component_prompt = create_component_prompt(jr_text=jr_text)
            answer = call_llm_oa(
                prompt=component_prompt, oa_api_key=self.settings.oa_api_key
            )
            dict_answer = json.loads(answer)
            dict_answer["idx"] = idx
            all_components.append(JRDecomposed(**dict_answer))

        return all_components

    def chunk_jr(
        self,
        jr_parsed_text,
    ) -> list[JRDecomposed]:
        jr_components = self._get_jr_components(jr_parsed_text=jr_parsed_text)
        validated_jr_components = self._validate_jr_components(
            jr_decomposes=jr_components
        )
        return validated_jr_components

    def chunk_cv(self, text: dict | list) -> list:
        if isinstance(text, dict):
            return semantic_chunk(
                text_experience=text["Experience"], text_skills=text["Skills"]
            )

        else:
            raise Exception

    def _create_report(self, reports: list[dict]):
        save_report(
            reports=reports,
            save_path=self.config.training.evaluation.save_path,
            save_name=self.config.training.evaluation.save_name,
        )

    def generate_score(self, prepared_evidence: list[Evidence]):
        all_answer = []
        for evidence in prepared_evidence:
            prompt = create_score_prompt(
                query=evidence.query, components=evidence.component
            )
            print(prompt)
            print("\n")
            answers = call_llm_oa(prompt=prompt, oa_api_key=self.settings.oa_api_key)
            answers_dict = json.loads(answers)
            all_answer.append(Evaluation(**answers_dict))
        return all_answer

    def evaluate(
        self,
        cv_chunks: list,
        jr_chunks: list,
        distances: list,
        indices: list,
        answers: str,
    ) -> None:
        if self.config.training.evaluation.print_report:
            return print_evaluation_report(
                cv_chunks=cv_chunks,
                jr_chunks=jr_chunks,
                distances=distances,
                indices=indices,
                answers=answers,
            )

        if self.config.training.evaluation.save_report:
            reports = create_evaluation_report(
                cv_chunks=cv_chunks,
                jr_chunks=jr_chunks,
                distances=distances,
                indices=indices,
                answers=answers,
            )

            self._create_report(reports=reports)

    def retrieve_base(
        self, cv_embedding: np.ndarray, jr_embedding: list[JREmbed], cv_chunks: list
    ):
        search_result = faiss_ip_search(
            cv_embedding=cv_embedding,
            jr_embedding=jr_embedding,
            query_top_k=self.config.training.retrieval.query_top_k,
            component_top_k=self.config.training.retrieval.component_top_k,
        )
        retrieved_chunks = retrieve_base_chunk(
            search_result=search_result, cv_chunks=cv_chunks
        )
        return retrieved_chunks

    def parse_text_jr(self, text: str) -> list[str]:
        if self.config.training.chunking.jr.method == ChunkingMethod.CHUNKING_NLTK:
            return parse_text_nltk(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.jr.method == ChunkingMethod.CHUNKING_NL:
            return parse_text_nl(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        else:
            raise Exception("unknown JR chunking method")

    def parse_text_cv(self, text: str) -> list[str] | dict:
        if self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_NLTK:
            return parse_text_nltk(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_NL:
            return parse_text_nl(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_RE:
            return parse_text_regex(text=text)
        elif self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_RE_NL:
            return parse_text_regex_nl(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif (
            self.config.training.chunking.cv.method
            == ChunkingMethod.CHUNKING_STRUCTURED
        ):
            return parse_text_structured(text=text)
        else:
            raise Exception("unknown CV chunking method")

    def embed_chunks(
        self, cv_chunks: list[str], jr_decomposed: list[JRDecomposed]
    ) -> tuple[list[JREmbed], np.ndarray]:

        if self.config.training.embedding.device == EmbeddingMethod.EMBEDDING_CUDA:
            if self._check_cuda():
                return embed_chunk_cuda(
                    cv_chunk=cv_chunks,
                    batch_size=self.config.training.embedding.batch_size,
                    jr_decomposed=jr_decomposed,
                )
            else:
                raise Exception("cannot use cuda since its not supported")
        else:
            raise Exception("unknown embedding device")

    def _load_JR(self) -> str:
        return load_JR(
            file_name=self.config.training.jr.file_name,
            folder_path=self.config.training.jr.folder_path,
        )

    def _load_CV(self) -> str:
        return load_CV(
            file_name=self.config.training.cv.file_name,
            folder_path=self.config.training.cv.folder_path,
        )

    def _check_cuda(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    T = TypeVar("T", bound="TrainingPipeline")

    @classmethod
    def load_from_config(cls: Type[T], config: Config, setting: Env) -> T:
        return cls(config, setting)

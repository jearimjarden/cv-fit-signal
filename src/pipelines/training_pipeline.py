from typing import Type, TypeVar
import shutil
import numpy as np
from ..services.generator.llm_client import call_llm_oa
from ..services.augmentator.prompt_builder import build_prompt
from ..IO.CV_loader_creator import load_CV
from ..IO.JR_loader_creator import load_JR
from ..tools.schemas import (
    Config,
    ChunkingMethod,
    EmbeddingMethod,
    Env,
    RetrievalMethod,
)
from ..services.preprocess.chunker import chunk_cv_regex, chunk_text_nltk, chunk_text_nl
from ..services.preprocess.embedder import embed_chunk_cpu, embed_chunk_cuda
from ..services.retrieval.vector_search import faiss_ip_search, retrieve_top_k
from ..services.evaluator.evaluation import (
    create_evaluation_report,
    print_evaluation_report,
)


class TrainingPipeline:
    def __init__(self, config: Config, settings: Env) -> None:
        self.settings = settings
        self.config = config

    def run(self):
        jr_text = self._load_JR()
        cv_text = self._load_CV()

        jr_chunks = self.chunk_text_jr(jr_text)
        cv_chunks = self.chunk_text_cv(cv_text)

        jr_embedding = self.embed_chunks(chunks=jr_chunks)
        cv_embedding = self.embed_chunks(chunks=cv_chunks)

        distances, indices = self.search_vector(
            jr_embedding=jr_embedding, cv_embedding=cv_embedding
        )

        retrieved_chunks = retrieve_top_k(cv_chunks=cv_chunks, indices=indices)

        self.evaluate_report(
            cv_chunks=cv_chunks,
            jr_chunks=jr_chunks,
            distances=distances,
            indices=indices,
        )

        prompt = self.augment_prompt(
            jr_chunks=jr_chunks, retrieved_chunks=retrieved_chunks
        )
        answers = self.generate_anwer(prompt=prompt)
        print(answers)

    def generate_anwer(self, prompt: str):
        answers = call_llm_oa(prompt=prompt, oa_api_key=self.settings.oa_api_key)
        return answers

    def augment_prompt(self, jr_chunks: list, retrieved_chunks: list):
        return build_prompt(jr_chunks=jr_chunks, cv_chunks=retrieved_chunks)

    def evaluate_report(
        self, cv_chunks: list, jr_chunks: list, distances: list, indices: list
    ) -> None:
        if self.config.training.evaluation.print_report:
            print_evaluation_report(
                cv_chunks=cv_chunks,
                jr_chunks=jr_chunks,
                distances=distances,
                indices=indices,
            )

        if self.config.training.evaluation.save_report:
            create_evaluation_report(
                cv_chunks=cv_chunks,
                jr_chunks=jr_chunks,
                distances=distances,
                indices=indices,
            )

    def search_vector(
        self, jr_embedding: np.ndarray, cv_embedding: np.ndarray
    ) -> tuple[list, list]:
        if self.config.training.retrieval.method == RetrievalMethod.RetrievalFaissIP:
            return faiss_ip_search(
                jr_embedding=jr_embedding,
                cv_embedding=cv_embedding,
                top_k=self.config.training.retrieval.top_k,
            )
        else:
            raise Exception

    def chunk_text_jr(self, text: str) -> list[str]:
        if self.config.training.chunking.jr.method == ChunkingMethod.CHUNKING_NLTK:
            return chunk_text_nltk(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.jr.method == ChunkingMethod.CHUNKING_NL:
            return chunk_text_nl(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        else:
            raise Exception("unknown JR chunking method")

    def chunk_text_cv(self, text: str) -> list[str]:
        if self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_NLTK:
            return chunk_text_nltk(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_NL:
            return chunk_text_nl(
                text=text,
                chunk_size=self.config.training.chunking.cv.chunk_size,
                stride=self.config.training.chunking.cv.stride,
            )
        elif self.config.training.chunking.cv.method == ChunkingMethod.CHUNKING_RE:
            return chunk_cv_regex(text=text)
        else:
            raise Exception("unknown CV chunking method")

    def embed_chunks(self, chunks: list[str]):
        if self.config.training.embedding.device == EmbeddingMethod.EMBEDDING_CPU:
            return embed_chunk_cpu(
                chunks=chunks, batch_size=self.config.training.embedding.batch_size
            )
        if self.config.training.embedding.device == EmbeddingMethod.EMBEDDING_CUDA:
            if self._check_cuda():
                return embed_chunk_cuda(
                    chunks=chunks, batch_size=self.config.training.embedding.batch_size
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
    def load_from_config(cls: Type[T], config: Config, setting=Env) -> T:
        return cls(config, setting)

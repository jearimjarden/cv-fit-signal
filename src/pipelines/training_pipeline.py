from typing import Type, TypeVar
import shutil
import numpy as np
from ..IO.CV_loader_creator import load_CV
from ..IO.JR_loader_creator import load_JR
from ..tools.schemas import Config, ChunkingMethod, EmbeddingMethod, RetrievalMethod
from ..services.preprocess.chunker import chunk_cv_regex, chunk_text_nltk, chunk_text_nl
from ..services.preprocess.embedder import embed_chunk_cpu, embed_chunk_cuda
from ..services.retrieval.vector_search import faiss_ip_search


class TrainingPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self):
        jr_text = self._load_JR()
        cv_text = self._load_CV()

        jr_chunk = self.chunk_text_jr(jr_text)
        cv_chunk = self.chunk_text_cv(cv_text)

        print(jr_chunk)
        print(cv_chunk)

        jr_embedding = self.embed_chunks(chunks=jr_chunk)
        cv_embedding = self.embed_chunks(chunks=cv_chunk)

        print(jr_embedding.shape)
        print(cv_embedding.shape)

        distances, indices = self.vector_search(
            jr_embedding=jr_embedding, cv_embedding=cv_embedding
        )
        print(distances)
        print(indices)

    def vector_search(
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
    def load_from_config(cls: Type[T], config: Config) -> T:
        return cls(config)

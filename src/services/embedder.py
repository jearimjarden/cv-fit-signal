from sentence_transformers import SentenceTransformer
from ..tools.schemas import (
    CVChunk,
    CVEmbedding,
    EmbeddingDevice,
    JRChunks,
    JREmbedding,
    EmbeddingModel,
)
import shutil


class EmbeddingService:
    def __init__(
        self,
        model_name: EmbeddingModel = EmbeddingModel.MINI_LLM,
        device: EmbeddingDevice = EmbeddingDevice.CPU,
    ) -> None:
        cuda_available = self._check_cuda()
        if device.value == "cuda":
            if cuda_available:
                self.model = SentenceTransformer(model_name.value, device=device.value)
            elif not cuda_available:
                print("cuda not available using 'cpu' instead")
                self.model = SentenceTransformer(
                    model_name.value, device=EmbeddingDevice.CPU.value
                )
        elif device.value == "cpu":
            self.model = SentenceTransformer(model_name.value, device=device.value)

    def _check_cuda(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def embed_chunks(
        self,
        cv_chunks: list[CVChunk],
        batch_size: int,
        jr_chunks: list[JRChunks],
    ) -> tuple[list[JREmbedding], list[CVEmbedding]]:

        all_jr_embedding = []
        for jr_chunk in jr_chunks:
            jr_components_embedding = self.model.encode(
                jr_chunk.components, batch_size=batch_size, convert_to_numpy=True
            )
            jr_query_embedding = self.model.encode(
                [jr_chunk.job_requirement], batch_size=batch_size, convert_to_numpy=True
            )
            jr_embedding = JREmbedding(
                idx=jr_chunk.idx,
                job_requirement=jr_chunk.job_requirement,
                components=jr_chunk.components,
                components_embedding=jr_components_embedding,
                job_requirement_embedding=jr_query_embedding,
            )
            all_jr_embedding.append(jr_embedding)

        all_cv_embedding = []
        for cv_chunk in cv_chunks:
            cv_embedding = self.model.encode(
                cv_chunk.chunk, batch_size=batch_size, convert_to_numpy=True
            )
            all_cv_embedding.append(
                CVEmbedding(
                    idx=cv_chunk.idx,
                    type=cv_chunk.type,
                    chunk=cv_chunk.chunk,
                    embedding=cv_embedding,
                )
            )

        return all_jr_embedding, all_cv_embedding

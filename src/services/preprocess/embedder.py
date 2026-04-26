from sentence_transformers import SentenceTransformer
import numpy as np


def embed_chunk_cuda(chunks: list[str], batch_size: int) -> np.ndarray:
    embedder_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
    )
    return embedder_model.encode(chunks, batch_size=batch_size, convert_to_numpy=True)


def embed_chunk_cpu(chunks: list[str], batch_size: int) -> np.ndarray:
    embedder_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    return embedder_model.encode(chunks, batch_size=batch_size, convert_to_numpy=True)

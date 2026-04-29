import faiss
import numpy as np


def faiss_ip_search(
    jr_embedding: np.ndarray, cv_embedding: np.ndarray, top_k: int
) -> tuple[list, list]:
    jr_embedding = jr_embedding.astype("float32")
    cv_embedding = cv_embedding.astype("float32")
    faiss.normalize_L2(jr_embedding)
    faiss.normalize_L2(cv_embedding)

    dim = cv_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cv_embedding)  # type: ignore

    distances, indices = index.search(jr_embedding, top_k)  # type: ignore
    return distances, indices


def retrieve_top_k(cv_chunks: list, indices: list) -> list[list]:
    retrieved_chunk = []

    for indice in indices:
        chunk = []
        for i in indice:
            chunk.extend([cv_chunks[i]])
        retrieved_chunk.append(chunk)

    return retrieved_chunk

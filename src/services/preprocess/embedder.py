from sentence_transformers import SentenceTransformer
import numpy as np

from ...tools.schemas import JRDecomposed, JREmbed


def embed_chunk_cuda(
    cv_chunk: list[str],
    batch_size: int,
    jr_decomposed: list[JRDecomposed],
) -> tuple[list[JREmbed], np.ndarray]:
    embedder_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
    )
    all_jr_embedding = []
    for jr_decom in jr_decomposed:
        jr_components_embedding = embedder_model.encode(
            jr_decom.components, batch_size=batch_size, convert_to_numpy=True
        )
        jr_query_embedding = embedder_model.encode(
            [jr_decom.job_requirement], batch_size=batch_size, convert_to_numpy=True
        )
        jr_embedding = JREmbed(
            idx=jr_decom.idx,
            job_requirement=jr_decom.job_requirement,
            components=jr_decom.components,
            components_embedding=jr_components_embedding,
            job_requirement_embedding=jr_query_embedding,
        )
        all_jr_embedding.append(jr_embedding)

    cv_embedding = embedder_model.encode(
        cv_chunk, batch_size=batch_size, convert_to_numpy=True
    )
    return all_jr_embedding, cv_embedding


def embed_chunk_cpu(chunks: list[str], batch_size: int) -> np.ndarray:
    embedder_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    return embedder_model.encode(chunks, batch_size=batch_size, convert_to_numpy=True)

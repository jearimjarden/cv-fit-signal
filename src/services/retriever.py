import faiss
import numpy as np
from ..tools.schemas import (
    BaseRetrieval,
    BaseRetrievalComponent,
    BaseRetrievalQuery,
    BaseSearch,
    BaseSearchComponents,
    BaseSearchQuery,
    CVEmbedding,
    JREmbedding,
)


def faiss_ip_search(
    cv_embedding: list[CVEmbedding],
    jr_embedding: list[JREmbedding],
    query_top_k: int,
    component_top_k: int,
) -> list[BaseSearch]:
    flat_cv_embedding = []
    for embedding in cv_embedding:
        flat_cv_embedding.append(embedding.embedding)

    flat_cv_embedding = np.array(flat_cv_embedding, dtype=np.float32)

    dim = flat_cv_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(flat_cv_embedding)  # type: ignore

    all_base_search = []
    for jr in jr_embedding:
        components_retrieval = []

        for idx, component in enumerate(jr.components):
            component_embedding = jr.components_embedding[idx].astype("float32")
            component_embedding = component_embedding.reshape(1, -1)
            faiss.normalize_L2(component_embedding)

            distances, indices = index.search(component_embedding, component_top_k)  # type: ignore
            components_retrieval.append(
                BaseSearchComponents(
                    component=component, distances=distances[0], indices=indices[0]
                )
            )

        distances, indices = index.search(jr.job_requirement_embedding, query_top_k)  # type: ignore
        query_retrieval = BaseSearchQuery(
            query=jr.job_requirement, distances=distances[0], indices=indices[0]
        )

        all_base_search.append(
            BaseSearch(
                idx=jr.idx,
                query_search=query_retrieval,
                components_search=components_retrieval,
            )
        )

    return all_base_search


def retrieve_base_chunk(
    search_result: list[BaseSearch],
    idx_to_chunk: dict[int, str],
) -> list[BaseRetrieval]:
    all_retrieved_chunks = []

    for result in search_result:
        query_retrieved_chunks = [idx_to_chunk[x] for x in result.query_search.indices]
        retrieved_query = BaseRetrievalQuery(
            query=result.query_search.query,
            distances=result.query_search.distances,
            chunks=query_retrieved_chunks,
        )

        retrieved_components = []
        for component in result.components_search:
            components_retrieved_chunks = [idx_to_chunk[x] for x in component.indices]
            retrieved_components.append(
                BaseRetrievalComponent(
                    component=component.component,
                    distances=component.distances,
                    chunks=components_retrieved_chunks,
                )
            )

        all_retrieved_chunks.append(
            BaseRetrieval(
                idx=result.idx,
                query_retrieval=retrieved_query,
                components_retrieval=retrieved_components,
            )
        )

    return all_retrieved_chunks

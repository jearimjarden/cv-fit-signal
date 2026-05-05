import faiss
import numpy as np
from ...tools.schemas import (
    BaseRetrieval,
    BaseRetrievalComponent,
    BaseRetrievalQuery,
    BaseSearch,
    BaseSearchComponents,
    BaseSearchQuery,
    JREmbed,
    Evidence,
    EvidenceComponent,
    EvidenceQuery,
)


def faiss_ip_search(
    cv_embedding: np.ndarray,
    jr_embedding: list[JREmbed],
    query_top_k: int,
    component_top_k: int,
) -> list[BaseSearch]:
    dim = cv_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(cv_embedding)  # type: ignore

    all_base_retrieval = []
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

        all_base_retrieval.append(
            BaseSearch(
                idx=jr.idx,
                query_search=query_retrieval,
                components_search=components_retrieval,
            )
        )

    return all_base_retrieval


def retrieve_base_chunk(
    search_result: list[BaseSearch],
    cv_chunks,
):
    all_retrieved_chunks = []
    for result in search_result:
        query_retrieved_chunks = [cv_chunks[x] for x in result.query_search.indices]
        retrieved_query = BaseRetrievalQuery(
            query=result.query_search.query,
            distances=result.query_search.distances,
            chunks=query_retrieved_chunks,
        )

        retrieved_components = []
        for component in result.components_search:
            components_retrieved_chunks = [cv_chunks[x] for x in component.indices]
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


def retrieve_top_k(cv_chunks: list, indices: list, jr_to_chunks: list):
    print(jr_to_chunks)
    last_idx = 0
    counter = 0
    retrieved_chunks = []
    all_retrieved_chunks = []
    for idx in range(0, len(jr_to_chunks)):
        print(f"index:{idx}")

        if jr_to_chunks[idx] != last_idx:
            counter = 0
            last_idx = jr_to_chunks[idx]
            all_retrieved_chunks.append(retrieved_chunks)
            retrieved_chunks = []

        if jr_to_chunks[idx] == last_idx:
            if counter == 0:
                chunks_idx = []
                for i in indices[idx]:
                    chunks_idx.append(i)
                counter += 1
                print(
                    f"Idx: {jr_to_chunks[idx]} Counter: {counter}, Chunks: {chunks_idx}"
                )

                retrieved_chunks.extend([cv_chunks[idx] for idx in chunks_idx[:3]])

            elif counter > 0:
                chunks_idx = []
                for i in indices[idx]:
                    chunks_idx.append(i)
                counter += 1
                print(
                    f"Idx: {jr_to_chunks[idx]} Counter: {counter}, Chunks: {chunks_idx}"
                )
                retrieved_chunks.extend([cv_chunks[idx] for idx in chunks_idx[:2]])

        if idx == (len(jr_to_chunks) - 1):
            all_retrieved_chunks.append(retrieved_chunks)

    return all_retrieved_chunks


def prepare_evidence(
    base_retrieval: list[BaseRetrieval], query_rerank: int, component_rerank: int
) -> list[Evidence]:
    all_reranked_retrieval = []
    for retrieval in base_retrieval:
        query_name = retrieval.query_retrieval.query
        query_evidence = retrieval.query_retrieval.chunks[:query_rerank]
        detailed_query_evidence = []
        for idx, distances in enumerate(
            retrieval.query_retrieval.distances[:query_rerank]
        ):
            detailed_query_evidence.append(f"{query_evidence[idx]}")

        all_component_evidence = []
        for component in retrieval.components_retrieval:
            component_name = component.component
            component_evidence = component.chunks[:component_rerank]
            component_distances = component.distances[:component_rerank]

            detail_component_evidence = []
            for idx, distances in enumerate(component_distances):
                detail_component_evidence.append(f"{component_evidence[idx]}")

            all_component_evidence.append(
                EvidenceComponent(
                    component=component_name,
                    evidence=detail_component_evidence,
                )
            )
        all_reranked_retrieval.append(
            Evidence(
                idx=retrieval.idx,
                query=EvidenceQuery(query=query_name, evidence=detailed_query_evidence),
                component=all_component_evidence,
            )
        )
    return all_reranked_retrieval

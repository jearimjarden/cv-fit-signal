def create_evaluation_report(
    cv_chunks: list, jr_chunks: list, distances: list, indices: list
) -> list:
    all_reports = []

    for idx_chunks, jr_chunk in enumerate(jr_chunks):
        report = {
            "query": jr_chunk,
            "retrieval": [
                (distances[idx_chunks][j], cv_chunks[indices[idx_chunks][j]])
                for j in range(len(indices[idx_chunks]))
            ],
        }

        all_reports.append(report)

    return all_reports


def print_evaluation_report(
    cv_chunks: list, jr_chunks: list, distances: list, indices: list
):

    for idx_chunks, jr_chunk in enumerate(jr_chunks):
        print(f"Job Requirement: {jr_chunk}")

        for idx_indices, score in enumerate(indices[idx_chunks]):
            print(f"{idx_indices+1}. Score:{distances[idx_chunks][idx_indices]:.3f}")
            print(cv_chunks[score])
            print("\n")

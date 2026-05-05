import json


def create_evaluation_report(
    cv_chunks: list, jr_chunks: list, distances: list, indices: list, answers: str
) -> list[dict]:
    reports = []
    dict_answers = json.loads(answers)
    for idx_chunks, jr_chunk in enumerate(jr_chunks):
        row = {
            "query": jr_chunk,
            "score": dict_answers["results"][idx_chunks]["score"],
            "reason": dict_answers["results"][idx_chunks]["reason"],
            "evidence": dict_answers["results"][idx_chunks]["evidence"],
        }

        for j in range(len(indices[idx_chunks])):
            row[f"top{j+1}_score"] = str(round(distances[idx_chunks][j], 3))
            row[f"top{j+1}_text"] = cv_chunks[indices[idx_chunks][j]]

        reports.append(row)

    return reports


def print_evaluation_report(
    cv_chunks: list, jr_chunks: list, distances: list, indices: list, answers: list
):

    dict_answers = [json.loads(answer) for answer in answers]

    for idx_chunks, jr_chunk in enumerate(jr_chunks):
        print(f"Job Requirement: {jr_chunk}")
        # for key, answer in dict_answers["results"][idx_chunks].items():
        #     print(f"{key}: {answer}")
        print("answer")
        print(dict_answers[idx_chunks])
        print("\nTop-K\n")
        for idx_indices, score in enumerate(indices[idx_chunks]):
            print(f"{idx_indices+1}. Score:{distances[idx_chunks][idx_indices]:.3f}")
            print(cv_chunks[score])
            print("\n")

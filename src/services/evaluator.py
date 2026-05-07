from ..tools.schemas import (
    Capability,
    ConfigEvaluation,
    Evidence,
    Evaluation,
    Score,
    ReportScore,
    Report,
    BaseRetrieval,
    EvidenceComponent,
    EvidenceQuery,
)
from .llm_client import LLMClient
from .prompt_builder import create_score_prompt, create_report_prompt
import json
from datetime import datetime


class EvaluatorService:
    def __init__(self, llm_client: LLMClient, evaluation: ConfigEvaluation) -> None:
        self.llm_client = llm_client
        self.evaluation = evaluation

    def generate_evaluation(
        self,
        base_retrieval: list[BaseRetrieval],
        query_rerank: int,
        component_rerank: int,
    ) -> list[Evaluation]:

        prepared_evidence = self._prepare_evidence(
            base_retrieval=base_retrieval,
            query_rerank=query_rerank,
            component_rerank=component_rerank,
        )
        all_response = []
        for evidence in prepared_evidence:
            prompt = create_score_prompt(
                query=evidence.query, components=evidence.component
            )
            response = self.llm_client.generate(prompt=prompt)
            print(response)
            response_dict = json.loads(response)
            response_dict["query"] = evidence.query.query
            all_response.append(Evaluation(**response_dict))
        return all_response

    def _prepare_evidence(
        self,
        base_retrieval: list[BaseRetrieval],
        query_rerank: int,
        component_rerank: int,
    ) -> list[Evidence]:
        all_reranked_retrieval = []

        for retrieval in base_retrieval:
            query_name = retrieval.query_retrieval.query
            query_evidence = retrieval.query_retrieval.chunks[:query_rerank]
            detailed_query_evidence = []
            for idx in range(0, len(query_evidence), 1):
                detailed_query_evidence.append(f"{query_evidence[idx]}")

            all_component_evidence = []
            for component in retrieval.components_retrieval:
                component_name = component.component
                component_evidence = component.chunks[:component_rerank]

                detail_component_evidence = []
                for idx in range(0, len(component_evidence), 1):
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
                    query=EvidenceQuery(
                        query=query_name, evidence=detailed_query_evidence
                    ),
                    component=all_component_evidence,
                )
            )

        return all_reranked_retrieval

    def _calculate_score(
        self,
        evidence_score: float,
        capability_level: Capability,
        responsibility_multiplier: float,
    ) -> float:

        capability_score = capability_level.weight()

        score = (
            (evidence_score * self.evaluation.evidence_score_mul)
            * (capability_score * self.evaluation.capability_score_mul)
            * (
                responsibility_multiplier
                * self.evaluation.responsibility_multiplier_mul
            )
        )

        return score

    def generate_score(self, evaluations: list[Evaluation]) -> list[Score]:
        all_scoring = []
        for evaluation in evaluations:
            jr_name = evaluation.query
            all_reason = []
            all_score = []

            for result in evaluation.result:
                score = self._calculate_score(
                    evidence_score=result.evidence_score,
                    capability_level=Capability(
                        capability_level=result.capability_level  # type: ignore
                    ),
                    responsibility_multiplier=result.responsible_multiplier,
                )
                all_reason.append(result.reason)
                all_score.append(score)

            final_score = sum(all_score) / len(evaluation.result)
            all_scoring.append(
                Score(query=jr_name, final_score=final_score, reason=all_reason)
            )

        return all_scoring

    def generate_report(self, score: list[Score], candidate_name: str) -> Report:
        prompt = create_report_prompt(scoring=score)
        response = self.llm_client.generate(prompt=prompt)
        dict_answer = json.loads(response)

        final_score = []
        all_report = []
        for idx, answer in enumerate(dict_answer["result"]):
            final_score.append(score[idx].final_score)
            all_report.append(
                ReportScore(
                    query=score[idx].query,
                    score=round(score[idx].final_score, 3),
                    reason=answer,
                )
            )
        return Report(
            datetime=datetime.now().strftime("%d-%m-%Y %H:%M"),
            name=candidate_name,
            report=all_report,
            final_score=round((sum(final_score)) / len(final_score), 3),
        )

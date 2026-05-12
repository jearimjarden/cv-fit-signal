import logging
import json
from datetime import datetime
from .llm_client import LLMClient
from .prompt_builder import create_score_prompt, create_report_prompt
from ..tools.schemas import (
    Capability,
    ConfigEvaluation,
    Evidence,
    Evaluation,
    InferenceStage,
    ReportInput,
    Score,
    ReportScore,
    Report,
    BaseRetrieval,
    EvidenceComponent,
    EvidenceQuery,
)
from ..tools.observabillity import LatencyStore, track_latency

logger = logging.getLogger(__name__)


class EvaluatorService:
    def __init__(
        self,
        llm_client: LLMClient,
        evaluation: ConfigEvaluation,
        latency_store: LatencyStore,
    ) -> None:
        self.llm_client = llm_client
        self.evaluation = evaluation
        self.latency_store = latency_store

    @track_latency(stage=InferenceStage.EVALUATION)
    def generate_evaluation(
        self,
        base_retrieval: list[BaseRetrieval],
    ) -> list[Evaluation]:
        prepared_evidence = self._prepare_evidence(base_retrieval=base_retrieval)

        all_response = []
        for evidence in prepared_evidence:
            prompt = create_score_prompt(
                query=evidence.query, components=evidence.component
            )

            try:
                response = self.llm_client.generate(
                    prompt=prompt, stage=InferenceStage.EVALUATION
                )
                response_dict = json.loads(response)
                response_dict["query"] = evidence.query.query
                all_response.append(Evaluation(**response_dict))

            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON output detected, attempting JSON repair",
                    extra={"stage": InferenceStage.EVALUATION},
                )
                fix_response = self.llm_client.json_repair(context=response)
                fix_response["query"] = evidence.query.query
                all_response.append(Evaluation(**fix_response))

        logger.debug(
            "Evaluation generated",
            extra={"stage": InferenceStage.EVALUATION, "result": all_response},
        )
        return all_response

    def _prepare_evidence(
        self,
        base_retrieval: list[BaseRetrieval],
    ) -> list[Evidence]:
        all_reranked_retrieval = []

        for retrieval in base_retrieval:
            query_name = retrieval.query_retrieval.query
            query_evidence = retrieval.query_retrieval.chunks
            detailed_query_evidence = []

            if len(query_evidence) == 0:
                detailed_query_evidence.append("No Global Evidence")

            elif len(query_evidence) > 0:
                for idx in range(0, len(query_evidence), 1):
                    detailed_query_evidence.append(f"{query_evidence[idx]}")

            all_component_evidence = []
            for component in retrieval.components_retrieval:
                component_name = component.component
                component_evidence = component.chunks

                detail_component_evidence = []
                if len(component_evidence) == 0:
                    detail_component_evidence.append("No Component Evidence")

                elif len(component_evidence) > 0:
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
        logger.debug(
            "Evidence prepared",
            extra={
                "stage": InferenceStage.EVALUATION,
                "result": all_reranked_retrieval,
            },
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
            (evidence_score * self.evaluation.evidence_mul)
            * (capability_score * self.evaluation.capability_mul)
            * (responsibility_multiplier * self.evaluation.responsibility_mul)
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

            summed_score = sum(all_score)

            if summed_score != 0:
                final_score = round(summed_score / len(all_score), 3)
            else:
                final_score = 0.0

            all_scoring.append(
                Score(query=jr_name, score=final_score, reason=all_reason)
            )

        logger.debug(
            "Score generated",
            extra={"stage": InferenceStage.SCORING, "result": all_scoring},
        )

        return all_scoring

    @track_latency(stage=InferenceStage.REPORT)
    def generate_report(self, scores: list[Score], candidate_name: str) -> Report:
        prompt = create_report_prompt(scoring=scores)

        try:
            response = self.llm_client.generate(
                prompt=prompt, stage=InferenceStage.REPORT
            )
            dict_answer = json.loads(response)

        except json.JSONDecodeError:
            logger.warning(
                "Invalid JSON output detected, attempting JSON repair",
                extra={"stage": InferenceStage.REPORT},
            )
            dict_answer = self.llm_client.json_repair(context=response)

        validated_anwer = ReportInput(**dict_answer)

        if len(scores) != len(validated_anwer.result):
            logger.warning(
                "Report generation failed to match the number of reason",
                extra={
                    "stage": InferenceStage.REPORT,
                    "n_reason": len(scores),
                    "report_n": len(validated_anwer.result),
                },
            )

        final_score = []
        all_report = []
        for idx, score in enumerate(scores):
            final_score.append(score.score)
            all_report.append(
                ReportScore(
                    query=score.query,
                    score=score.score,
                    reason=validated_anwer.result[idx],
                )
            )

        summed_score = sum(final_score)

        if summed_score != 0:
            report_score = summed_score / len(final_score)
        else:
            report_score = 0.0

        final_report = Report(
            datetime=datetime.now().strftime("%d-%m-%Y_%H:%M"),
            name=candidate_name,
            report=all_report,
            report_score=report_score,
        )

        logger.debug(
            "Report generated",
            extra={"stage": InferenceStage.REPORT, "result": final_report},
        )

        return final_report

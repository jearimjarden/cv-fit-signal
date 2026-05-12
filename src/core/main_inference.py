import os
import logging
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from ..pipelines.inference_pipeline import InferencePipeline
from ..tools.exceptions_schemas import ConfigurationError, LoggedError
from ..tools.observabillity import LatencyStore, TrackToken
from ..tools.logging_setup import setup_bootstrap_logger, setup_logger
from ..tools.config_loader import load_config, load_env
from ..tools.schemas import Config, Env, JRInput, PipelineStage

# Tester
test = {
    "text": "1. Strong proficiency in Python\n2. Ability to design and develop backend services\n3. Familiarity with containerization tools (e.g., Docker)\n4. Experience applying machine learning techniques to real problems\n5. Knowledge of relational databases and SQL"
}
test2 = {
    "text": "Develop scalable backend systems for production use\n2. Work with data-driven models and analytics pipelines\n- Collaborate using modern development tools and workflows\n4. Ensure system deployment and environment consistency\n5. Handle structured data storage and querying"
}


def main(logger: logging.Logger, config: Config, settings: Env) -> None:
    try:
        latency_store = LatencyStore()
        track_token = TrackToken(llm_config=config.llm)

        pipeline = InferencePipeline.load_from_config(
            config=config,
            setting=settings,
            latency_store=latency_store,
            track_token=track_token,
        )
        logger.info(
            "Inference Pipeline created",
            extra={
                "stage": PipelineStage.INFERENCE,
                "config": {
                    "input_mode": config.input_mode,
                    "cv_name": config.file_service.cv.file_name,
                    "jr_name": config.file_service.jr.file_name,
                    "query_top_k": config.retrieval.query_top_k,
                    "component_top_k": config.retrieval.component_top_k,
                    "retrieval_filter": config.retrieval.filter_below_threshold,
                    "evidence_mul": config.evaluation.evidence_mul,
                    "capability_mul": config.evaluation.capability_mul,
                    "responsibility_mul": config.evaluation.responsibility_mul,
                },
            },
        )

        report = pipeline.run(cv_selection="ardi_pratama", jr_input=JRInput(**test))
        # print(report)

        sys.exit(0)

    except LoggedError:
        logger.error(
            "Exiting Inference Pipeline", extra={"stage": PipelineStage.INFERENCE}
        )
        sys.exit(1)

    except Exception:
        logger.critical(
            "Unexpected error occured",
            exc_info=True,
            extra={"stage": PipelineStage.INFERENCE},
        )
        sys.exit(2)


if __name__ == "__main__":
    try:
        bootstrap_logger = setup_bootstrap_logger()

        config = load_config()
        env = load_env()
        setup_logger(
            level=config.logger.level,
            environment=env.environment,
            pipeline_name=PipelineStage.INFERENCE,
            save_log=config.logger.save_log,
        )

        bootstrap_logger.info(
            "Starting Inference Pipeline",
            extra={
                "environment": env.environment,
                "logging_level": config.logger.level.value,
            },
        )
        logger = logging.getLogger(__name__)

        main(logger=logger, config=config, settings=env)

    except ConfigurationError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception:
        bootstrap_logger.critical("Unexpected error occured", exc_info=True)
        sys.exit(2)

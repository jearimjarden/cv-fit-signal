from ..tools.logging_setup import setup_bootstrap
from ..tools.config_loader import load_config, load_env
from ..pipelines.training_pipeline import TrainingPipeline
from ..tools.schemas import Config
import logging


def main(logger: logging.Logger, config: Config):
    pipeline = TrainingPipeline.load_from_config(config)
    pipeline.run()


if __name__ == "__main__":
    bootstrap_logger = setup_bootstrap()
    config = load_config()
    env = load_env()
    main(logger=bootstrap_logger, config=config)

from ..tools.logging_setup import setup_bootstrap
from ..tools.config_loader import load_config, load_env
from ..pipelines.inference_pipeline import InferencePipeline
from ..tools.schemas import Config, Env
import logging


def main(logger: logging.Logger, config: Config, settings: Env):
    pipeline = InferencePipeline.load_from_config(config=config, setting=settings)
    pipeline.run()


if __name__ == "__main__":
    bootstrap_logger = setup_bootstrap()
    config = load_config()
    env = load_env()
    main(logger=bootstrap_logger, config=config, settings=env)

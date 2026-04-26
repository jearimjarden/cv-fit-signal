import yaml
from pathlib import Path
from .schemas import Config, Env
import os


def load_config():
    path = Path("config.yaml")

    if not path.exists():
        raise Exception("config path is not exist")

    with open(path, "r") as file:
        config_data = yaml.safe_load(file)

    return Config(**config_data)


def load_env():
    env = Env()  # type: ignore
    os.environ["HF_TOKEN"] = env.hf_api_key
    return env

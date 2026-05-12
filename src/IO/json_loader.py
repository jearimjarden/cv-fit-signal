from pathlib import Path
import json
from ..tools.exceptions_schemas import InvalidFileError


def load_json(file_name: str, folder_path: str) -> dict:
    path = Path(folder_path) / file_name

    try:
        with open(path, "r") as file:
            return json.load(file)

    except FileNotFoundError as e:
        raise InvalidFileError(f"File was not found at {path}") from e

    except json.JSONDecodeError as e:
        raise InvalidFileError(f"Failed to decode json file for {path}") from e

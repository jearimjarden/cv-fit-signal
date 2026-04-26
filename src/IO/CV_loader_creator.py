from pathlib import Path
import json


def save_CV(file_name: str, folder_path: str, input_text: str):
    # check the validity of file_name
    path = Path(folder_path)

    if not path.exists():
        raise Exception("path is not exist")

    with open(path / file_name, "w") as file:
        json.dump(input_text, file, indent=2)


def load_CV(file_name: str, folder_path) -> str:
    path = Path(folder_path) / file_name

    if not path.exists():
        raise Exception("CV path is not exist")

    with open(path, "r") as file:
        data = json.load(file)

    return data

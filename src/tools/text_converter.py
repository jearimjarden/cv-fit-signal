import json


def convert_text_to_json(text_input: str) -> str:
    # check if the input is not None
    text_json = json.dumps(text_input, indent=2)

    return text_json

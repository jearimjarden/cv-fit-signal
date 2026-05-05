import re
from nltk.tokenize import sent_tokenize


# stride cannot be bigger than chunk_size
def parse_text_nltk(text: str, chunk_size: int, stride: int) -> list[str]:
    sentences = sent_tokenize(text)
    all_chunks = []

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    return all_chunks


def parse_text_nl(text: str, chunk_size: int, stride: int) -> list[str]:
    sentences = text.split("\n")

    all_chunks = []
    if len(sentences) < chunk_size:
        return sentences

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    all_chunks = [x for x in all_chunks if x]
    return all_chunks


def parse_text_regex(text: str):
    # this is a hardcoded chunker for semi structured CV
    all_chunks = []
    result = re.findall(r"(\w+):\s*\n([\S\s]+?)(?:\n\n|$)", text)
    for field, value in result:
        all_chunks.append(f"{field}: {value}")
    return all_chunks


def parse_text_regex_nl(text: str, chunk_size: int, stride: int):
    regex_chunk = parse_text_regex(text)
    all_chunk = []

    for chunk in regex_chunk:
        nl_chunk = parse_text_nl(text=chunk, chunk_size=chunk_size, stride=stride)
        all_chunk.extend(nl_chunk)

    return all_chunk


def parse_text_structured(text: str) -> dict:
    pattern = r"(Name|Summary|Skills|Experience|Education):\s*\n"
    splits = re.split(pattern, text)

    all_split = {}
    for x in range(1, len(splits), 2):
        key = splits[x]
        value = splits[x + 1]
        all_split[key] = value

    return all_split

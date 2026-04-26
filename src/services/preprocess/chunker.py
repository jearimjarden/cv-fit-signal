import re
from nltk.tokenize import sent_tokenize


# stride cannot be bigger than chunk_size
def chunk_text_nltk(text: str, chunk_size: int, stride: int) -> list[str]:
    sentences = sent_tokenize(text)
    all_chunks = []

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    return all_chunks


def chunk_text_nl(text: str, chunk_size: int, stride: int) -> list[str]:
    sentences = text.split("\n")

    all_chunks = []

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    return all_chunks


def chunk_cv_regex(text: str):
    # this is a hardcoded chunker for semi structured CV
    all_chunks = []
    result = re.findall(r"(\w+):\s*\n([\S\s]+?)(?:\n\n|$)", text)
    for field, value in result:
        all_chunks.append(f"{field}: {value}")
    return all_chunks

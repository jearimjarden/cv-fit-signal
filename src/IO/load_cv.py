from pathlib import Path
import json
import numpy as np
import re
from pydantic import ValidationError
from ..tools.exceptions_schemas import InvalidArtifact
from ..tools.schemas import CVChunk, CVEmbedding, Metadata


def load_cv_all(cv_selection: str) -> tuple[Metadata, list[CVChunk], list[CVEmbedding]]:
    load_path = Path("storage/candidates") / cv_selection
    if not load_path.exists():
        raise InvalidArtifact(f"CV data for '{cv_selection}' not found")

    try:
        metadata = _load_cv_metadata(load_path=load_path)
        cv_chunks = _load_cv_chunks(load_path=load_path)
        cv_embedding = _load_cv_embedding(load_path=load_path)

        if metadata.cv_embedding_n != metadata.cv_chunked_n:
            raise InvalidArtifact(
                "Invalid Metadata (cv_chunked_n and cv_embedding_n does not match)"
            )

        if metadata.cv_embedding_n != len(cv_embedding):
            raise InvalidArtifact("Metadata and cv_embedding does not match)")

        if len(cv_chunks) != len(cv_embedding):
            raise InvalidArtifact(
                f"CV chunks and embedding does not match (cv_chunks_n: {len(cv_chunks)}, cv_embedding_n: {len(cv_embedding)})"
            )

        return metadata, cv_chunks, cv_embedding

    except FileNotFoundError as e:
        match = re.search(r"'([^']+)'", str(e))

        if match:
            raise InvalidArtifact(f"File was not found for '{match.group(1)}'")

        else:
            raise InvalidArtifact(str(e))

    except TypeError as e:
        raise InvalidArtifact(f"Invalid metadata schema: {e}") from e

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(f"Missing artifact parameter: '{field}'")

            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden extra artifact parameter: '{field}'")

            else:
                messages.append(f"Invalid artifact value for '{field}': {err['msg']}")

        raise InvalidArtifact(" | ".join(messages)) from e


def _load_cv_metadata(load_path: Path) -> Metadata:
    with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

        return Metadata(**data)


def _load_cv_chunks(load_path: Path) -> list[CVChunk]:
    with open(load_path / "cv_chunks.json", "r", encoding="utf-8") as f:
        datas = json.load(f)

    cv_chunks = [CVChunk(**data) for data in datas]
    return cv_chunks


def _load_cv_embedding(load_path: Path) -> list[CVEmbedding]:
    loaded_embeddings = np.load(load_path / "cv_embedding.npy")
    cv_embedding = []

    for idx, embedding in enumerate(loaded_embeddings):
        cv_embedding.append(CVEmbedding(idx=idx, embedding=embedding))

    return cv_embedding

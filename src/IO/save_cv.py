from pathlib import Path
import json
import numpy as np
from dataclasses import asdict
from datetime import datetime
import logging
from ..tools.schemas import (
    LatencyStored,
    PreprocessStage,
    StructuredCV,
    CVChunk,
    CVEmbedding,
    TokenSummary,
)

logger = logging.getLogger(__name__)


def save_all_cv(
    cv_parsed: StructuredCV,
    cv_chunk: list[CVChunk],
    cv_embedding: list[CVEmbedding],
    cv_name_str: str,
    token_summary: TokenSummary,
    latency_stored: LatencyStored,
) -> None:
    storage_dir = Path("storage")
    storage_dir.mkdir(exist_ok=True)

    candidates_dir = storage_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)

    candidate_dir = candidates_dir / cv_name_str

    if candidate_dir.exists():
        logger.warning(
            f"Rewriting existed CV for '{cv_name_str}'",
            extra={"stage": PreprocessStage.SAVE},
        )

    candidate_dir.mkdir(exist_ok=True)

    _save_cv_parsed(cv_parsed=cv_parsed, save_path=candidate_dir)
    _save_cv_chunk(cv_chunks=cv_chunk, save_path=candidate_dir)
    cv_embedding_n = _save_cv_embedding(
        cv_embedding=cv_embedding, save_path=candidate_dir
    )
    _save_cv_metadata(
        name=cv_parsed.person_name,
        cv_embedding_n=cv_embedding_n,
        cv_parsed_n=len(cv_parsed),
        cv_chunks_n=len(cv_chunk),
        token_summary=token_summary,
        save_path=candidate_dir,
        latency_stored=latency_stored,
    )

    logger.debug(
        f"CV artifacts saved for '{cv_name_str}'",
        extra={"stage": PreprocessStage.SAVE, "save_path": candidate_dir},
    )


def _save_cv_parsed(cv_parsed: StructuredCV, save_path: Path) -> None:
    data = cv_parsed.model_dump()
    save_file = save_path / "cv_parsed.json"

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.debug(
        "Parsed CV saved", extra={"stage": PreprocessStage.SAVE, "save_path": save_file}
    )


def _save_cv_chunk(cv_chunks: list[CVChunk], save_path: Path) -> None:
    data = [chunk.model_dump() for chunk in cv_chunks]
    save_file = save_path / "cv_chunks.json"

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.debug(
        "CV chunk saved", extra={"stage": PreprocessStage.SAVE, "save_path": save_file}
    )


def _save_cv_embedding(cv_embedding: list[CVEmbedding], save_path: Path) -> int:
    data = [chunk.embedding for chunk in cv_embedding]
    save_file = save_path / "cv_embedding.npy"

    np.save(save_file, data)

    logger.debug(
        "CV embedding saved",
        extra={"stage": PreprocessStage.SAVE, "save_path": save_file},
    )

    return len(data)


def _save_cv_metadata(
    name: str,
    cv_embedding_n: int,
    cv_parsed_n: int,
    cv_chunks_n: int,
    token_summary: TokenSummary,
    save_path: Path,
    latency_stored: LatencyStored,
) -> None:
    data = {
        "cv_name": name,
        "created_date": datetime.now().strftime("%d-%m-%Y_%H:%M"),
        "cv_parsed_n": cv_parsed_n,
        "cv_chunked_n": cv_chunks_n,
        "cv_embedding_n": cv_embedding_n,
        "token_summary": asdict(token_summary),
        "latencies_ms": asdict(latency_stored),
    }
    save_file = save_path / "metadata.json"

    with open(save_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.debug(
        "CV metadata saved",
        extra={"stage": PreprocessStage.SAVE, "save_path": save_file},
    )

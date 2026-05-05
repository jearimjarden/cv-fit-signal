from pydantic import BaseModel, Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
from enum import Enum
import numpy as np


class Env(BaseSettings):
    environment: str = Field(...)
    oa_api_key: str = Field(...)
    hf_api_key: str = Field(...)
    model_config = SettingsConfigDict(extra="forbid", env_file=".env")


class ChunkingMethod(str, Enum):
    CHUNKING_NLTK = "nltk"
    CHUNKING_NL = "nl"
    CHUNKING_RE = "re"
    CHUNKING_RE_NL = "re_nl"
    CHUNKING_STRUCTURED = "structured"


class RetrievalMethod(str, Enum):
    RetrievalFaissIP = "faiss_ip"


class EmbeddingMethod(str, Enum):
    EMBEDDING_CUDA = "cuda"
    EMBEDDING_CPU = "cpu"


class ConfigTrainingCV(BaseModel):
    file_name: str = Field(...)
    folder_path: str = Field(...)


class ConfigTrainingJR(BaseModel):
    file_name: str = Field(...)
    folder_path: str = Field(...)


class ConfigTrainingChunkingOption(BaseModel):
    method: ChunkingMethod = Field(...)
    chunk_size: int = Field(..., ge=1)
    stride: int = Field(..., ge=1)


class ConfigTrainingChunking(BaseModel):
    cv: ConfigTrainingChunkingOption = Field(...)
    jr: ConfigTrainingChunkingOption = Field(...)


class ConfigTrainingEmbedding(BaseModel):
    device: EmbeddingMethod = Field(...)
    batch_size: int = Field(4096, ge=4, le=16384)


class ConfigTrainingRetrieval(BaseModel):
    method: RetrievalMethod = Field(...)
    query_top_k: int = Field(...)
    component_top_k: int = Field(...)
    query_rerank: int = Field(...)
    component_rerank: int = Field(...)


class ConfigTrainingEvaluation(BaseModel):
    print_report: bool = Field(False)
    save_report: bool = Field(False)
    save_path: str = Field(...)
    save_name: str = Field(...)


class ConfigTraining(BaseModel):
    cv: ConfigTrainingCV = Field(...)
    jr: ConfigTrainingJR = Field(...)
    chunking: ConfigTrainingChunking = Field(...)
    embedding: ConfigTrainingEmbedding = Field(...)
    retrieval: ConfigTrainingRetrieval = Field(...)
    evaluation: ConfigTrainingEvaluation = Field(...)


class Config(BaseModel):
    training: ConfigTraining = Field(...)


class JRDecomposed(BaseModel):
    idx: int = Field(...)
    job_requirement: str = Field(...)
    components: list[str] = Field(...)
    reason: str = Field(...)


class JREmbed(BaseModel):
    idx: int = Field(...)
    job_requirement: str = Field(...)
    components: list[str] = Field(...)
    job_requirement_embedding: np.ndarray = Field(...)
    components_embedding: np.ndarray = Field(...)

    class Config:
        arbitrary_types_allowed = True


class BaseSearchQuery(BaseModel):
    query: str = Field(...)
    distances: list = Field(...)
    indices: list = Field(...)


class BaseSearchComponents(BaseModel):
    component: str = Field(...)
    distances: list = Field(...)
    indices: list = Field(...)


class BaseSearch(BaseModel):
    idx: int = Field(...)
    query_search: BaseSearchQuery = Field(...)
    components_search: list[BaseSearchComponents] = Field(...)


class BaseRetrievalQuery(BaseModel):
    query: str = Field(...)
    distances: list = Field(...)
    chunks: list = Field(...)


class BaseRetrievalComponent(BaseModel):
    component: str = Field(...)
    distances: list = Field(...)
    chunks: list = Field(...)


class BaseRetrieval(BaseModel):
    idx: int = Field(...)
    query_retrieval: BaseRetrievalQuery = Field(...)
    components_retrieval: list[BaseRetrievalComponent] = Field(...)


class EvidenceComponent(BaseModel):
    component: str = Field(...)
    evidence: list = Field(...)


class EvidenceQuery(BaseModel):
    query: str = Field(...)
    evidence: list = Field(...)


class Evidence(BaseModel):
    idx: int = Field(...)
    query: EvidenceQuery = Field(...)
    component: list[EvidenceComponent] = Field(...)


class EvaluationResult(BaseModel):
    components: str = Field(...)
    evidence_score: float = Field(...)
    responsible_multiplier: float = Field(...)
    capability_level: str = Field(...)
    reason: str = Field(...)


class Evaluation(BaseModel):
    result: list[EvaluationResult] = Field(...)

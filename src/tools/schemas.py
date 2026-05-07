from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict, BaseSettings
from enum import Enum
import numpy as np
from torch import Optional
from dataclasses import dataclass


class EmbeddingDevice(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class EmbeddingModel(str, Enum):
    MINI_LLM = "sentence-transformers/all-MiniLM-L6-v2"


class CapabilityLevel(str, Enum):
    EXPLICIT_STRONG = "explicit_strong"
    EXPLICIT_WEAK = "explicit_weak"
    IMPLICIT_STRONG = "implicit_strong"
    IMPLICIT_WEAK = "implicit_weak"
    MISSING = "missing"


@dataclass
class Capability:
    capability_level: CapabilityLevel

    def __post_init__(self):
        if isinstance(self.capability_level, str):
            self.capability_level = CapabilityLevel(self.capability_level)

    def weight(self) -> float:
        mapping = {
            CapabilityLevel.EXPLICIT_STRONG: 1.0,
            CapabilityLevel.EXPLICIT_WEAK: 0.8,
            CapabilityLevel.IMPLICIT_STRONG: 0.6,
            CapabilityLevel.IMPLICIT_WEAK: 0.4,
            CapabilityLevel.MISSING: 0.0,
        }

        return mapping[self.capability_level]


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


class CVInput(BaseModel):
    text: str = Field(...)


class ConfigFileServiceItem(BaseModel):
    file_name: str = Field(...)
    folder_path: str = Field(...)


class ConfigEmbedding(BaseModel):
    device: EmbeddingDevice = Field(...)
    batch_size: int = Field(4096, ge=4, le=16384)
    model: str = Field(...)


class ConfigFileService(BaseModel):
    cv: ConfigFileServiceItem = Field(...)
    jr: ConfigFileServiceItem = Field(...)


class ConfigRetrieval(BaseModel):
    query_top_k: int = Field(...)
    component_top_k: int = Field(...)


class ConfigEvaluation(BaseModel):
    evidence_score_mul: float = Field(1.0)
    capability_score_mul: float = Field(1.0)
    responsibility_multiplier_mul: float = Field(1.0)


class ConfigJRChunk(BaseModel):
    chunk_size: int = Field(...)
    stride: int = Field(...)


class Config(BaseModel):
    api_service: bool = Field(...)
    file_service: ConfigFileService = Field(...)
    jr_chunk: ConfigJRChunk = Field(...)
    embedding: ConfigEmbedding = Field(...)
    retrieval: ConfigRetrieval = Field(...)
    evaluation: ConfigEvaluation = Field(...)


class JRChunks(BaseModel):
    idx: int = Field(...)
    job_requirement: str = Field(...)
    components: list[str] = Field(...)
    reason: str = Field(...)


class JREmbedding(BaseModel):
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
    query: str = Field(...)
    result: list[EvaluationResult] = Field(...)


class StructuredCVItem(BaseModel):
    name: str = Field(...)
    item: list = Field(...)
    model_config = ConfigDict(extra="forbid")


class StructuredCVLanguage(BaseModel):
    name: str = Field(...)
    level: str = Field(...)


class StructuredCV(BaseModel):
    person_name: str = Field(...)
    education: list = Field(...)
    technical_skills: list[StructuredCVItem] = Field(...)
    work_experience: list[StructuredCVItem] = Field(...)
    project: list[StructuredCVItem] = Field(...)
    soft_skills: list = Field(...)
    languages: list[StructuredCVLanguage] = Field(...)

    model_config = ConfigDict(extra="forbid")


class CVChunk(BaseModel):
    idx: int = Field(...)
    type: str = Field(...)
    chunk: str = Field(...)


class CVEmbedding(BaseModel):
    idx: int = Field(...)
    type: str = Field(...)
    chunk: str = Field(...)
    embedding: np.ndarray = Field(...)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Score(BaseModel):
    query: str = Field(...)
    final_score: float = Field(...)
    reason: list[str] = Field(...)


class ReportScore(BaseModel):
    query: str = Field(...)
    score: float = Field(...)
    reason: str = Field(...)


class Report(BaseModel):
    datetime: str = Field(...)
    name: Optional[str] = Field(default=None)
    report: list[ReportScore] = Field(...)
    final_score: float = Field(...)

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict, BaseSettings
from enum import Enum


class Env(BaseSettings):
    environment: str = Field(...)
    oa_api_key: str = Field(...)
    hf_api_key: str = Field(...)
    model_config = SettingsConfigDict(extra="forbid", env_file=".env")


class ChunkingMethod(str, Enum):
    CHUNKING_NLTK = "nltk"
    CHUNKING_NL = "nl"
    CHUNKING_RE = "re"


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
    top_k: int = Field(...)


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

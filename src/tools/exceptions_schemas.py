from .schemas import InferenceStage, PreprocessStage, PipelineStage


class LoggedError(Exception):
    pass


# ===============================================================================
# Configuration Exceptions
# ===============================================================================
class ConfigurationError(Exception):
    def __init__(self, message, stage: str):
        self.stage = stage
        super().__init__(message)


class InvalidConfig(ConfigurationError):
    def __init__(self, message: str):
        super().__init__(message, stage="config")


class InvalidSettings(ConfigurationError):
    def __init__(self, message: str):
        super().__init__(message, stage="config")


# ===============================================================================


# ===============================================================================
# Preprocess Exceptions
# ===============================================================================
class PreprocessorError(Exception):
    def __init__(self, message: str, stage: str):
        self.stage = stage
        super().__init__(message)


class InvalidCVError(PreprocessorError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE)


class InvalidParsedCV(PreprocessorError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE)


# ===============================================================================


# ===============================================================================
# Inference Exceptions
# ===============================================================================
class InferenceError(Exception):
    def __init__(self, message: str, stage: str):
        self.stage = stage
        super().__init__(message)


class InvalidJRError(InferenceError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE)


class InvalidFileError(InferenceError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.FILEINPUT)


class InvalidArtifact(InferenceError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.ARTIFACT)


# ===============================================================================


# ===============================================================================
# LLM Client Exceptions
# ===============================================================================
class LLMError(Exception):
    def __init__(self, message: str, stage: str):
        self.stage = stage
        super().__init__(message)


class InvalidJSON(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM)


class InvalidResponse(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM)


class LLMTimeoutError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM)


class LLMConnectionError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM)


class LLMAuthenticationError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM)


# ===============================================================================

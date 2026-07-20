from __future__ import annotations

class InferenceError(RuntimeError):
    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause is not None:
            return f"{super().__str__()} | caused by: {self.cause!r}"
        return super().__str__()

class ModelLoadError(InferenceError):
    pass

class ImageValidationError(InferenceError):
    pass

class ImageNotFoundError(ImageValidationError):
    pass

class UnsupportedFormatError(ImageValidationError):
    pass

class CorruptedImageError(ImageValidationError):
    pass

class PreprocessingError(InferenceError):
    pass

class DeviceError(InferenceError):
    pass

class InferenceRuntimeError(InferenceError):
    pass

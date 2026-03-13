"""
Exception hierarchy for the chunker library.

This module defines all custom exceptions used throughout the chunker,
providing structured error handling with context information.
"""

from typing import Any, Dict, Optional


class ChunkerException(Exception):
    """
    Base exception for all chunker errors.
    
    All exceptions include contextual information to aid debugging:
    - component: Which component raised the error
    - operation: What operation was being performed
    - original_exception: The underlying exception (if wrapped)
    - context: Additional context dictionary
    """
    
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.original_exception = original_exception
        self.context = context or {}
        
    def __str__(self) -> str:
        """Format exception with context information."""
        msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg = f"{msg} (context: {context_str})"
        if self.original_exception:
            msg = f"{msg} [caused by: {type(self.original_exception).__name__}: {self.original_exception}]"
        return msg


class ConfigurationError(ChunkerException):
    """Raised when configuration is invalid."""
    pass


class InvalidConstraintError(ConfigurationError):
    """Raised when configuration constraints are violated (e.g., min > max)."""
    pass


class MissingRequirementError(ConfigurationError):
    """Raised when required configuration is missing (e.g., API key for strategy)."""
    pass


class IncompatibilityError(ConfigurationError):
    """Raised when configuration components are incompatible (e.g., wrong model for provider)."""
    pass


class InputValidationError(ChunkerException):
    """Raised when input data is invalid."""
    pass


class ProcessingError(ChunkerException):
    """Raised when processing fails during chunking."""
    pass


class EmbeddingError(ProcessingError):
    """Raised when embedding generation fails."""
    pass


class LanguageDetectionError(ProcessingError):
    """Raised when language detection fails."""
    pass


class SegmentationError(ProcessingError):
    """Raised when sentence segmentation fails."""
    pass


class ResourceError(ChunkerException):
    """Raised when resource loading or access fails."""
    pass


class ModelLoadError(ResourceError):
    """Raised when model loading fails."""
    pass


class DependencyMissingError(ResourceError):
    """Raised when optional dependency is missing."""
    pass


class CacheError(ResourceError):
    """Raised when cache operations fail."""
    pass

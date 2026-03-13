"""
Structured logging for chunker operations.

Provides structured logging with JSON output for production observability.
Logs key events, timing information, and fallback activations.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from chunker.config import StrategyType
from chunker.models import Language

__all__ = ["StructuredLogger"]


class StructuredLogger:
    """
    Structured logging for chunker operations.
    
    Provides structured logging with JSON output for production observability.
    Supports configurable log levels and structured event logging.
    
    Parameters
    ----------
    name : str
        Logger name (typically module name)
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    structured : bool
        If True, emit JSON-formatted structured logs. If False, use standard text format.
    
    Examples
    --------
    >>> logger = StructuredLogger("chunker", level=logging.INFO)
    >>> logger.log_chunking_start("doc1", 1000, StrategyType.SEMANTIC)
    >>> logger.log_chunking_complete("doc1", 5, 234.5, StrategyType.SEMANTIC)
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        structured: bool = True
    ):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : int
            Logging level
        structured : bool
            Whether to use structured JSON logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.structured = structured
        
        # Ensure logger has at least one handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            self.logger.addHandler(handler)
    
    def _log_structured(
        self,
        level: int,
        event: str,
        **kwargs: Any
    ) -> None:
        """
        Log a structured event.
        
        Parameters
        ----------
        level : int
            Logging level
        event : str
            Event name
        **kwargs
            Additional event data
        """
        if self.structured:
            log_entry = {
                "timestamp": time.time(),
                "event": event,
                **kwargs
            }
            self.logger.log(level, json.dumps(log_entry))
        else:
            # Format as readable text
            msg = f"[{event}] " + " ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.log(level, msg)
    
    def log_chunking_start(
        self,
        doc_id: str,
        text_length: int,
        strategy: StrategyType
    ) -> None:
        """
        Log start of chunking operation.
        
        Parameters
        ----------
        doc_id : str
            Document identifier
        text_length : int
            Length of text in characters
        strategy : StrategyType
            Chunking strategy being used
        """
        self._log_structured(
            logging.INFO,
            "chunking_start",
            component="SemanticChunker",
            doc_id=doc_id,
            text_length=text_length,
            strategy=strategy.value if isinstance(strategy, StrategyType) else str(strategy)
        )
    
    def log_chunking_complete(
        self,
        doc_id: str,
        chunk_count: int,
        duration_ms: float,
        strategy: StrategyType
    ) -> None:
        """
        Log completion of chunking operation.
        
        Parameters
        ----------
        doc_id : str
            Document identifier
        chunk_count : int
            Number of chunks created
        duration_ms : float
            Duration in milliseconds
        strategy : StrategyType
            Chunking strategy used
        """
        self._log_structured(
            logging.INFO,
            "chunking_complete",
            component="SemanticChunker",
            doc_id=doc_id,
            chunk_count=chunk_count,
            duration_ms=duration_ms,
            strategy=strategy.value if isinstance(strategy, StrategyType) else str(strategy)
        )
    
    def log_language_detected(
        self,
        doc_id: str,
        language: Language,
        confidence: float = 1.0
    ) -> None:
        """
        Log language detection result.
        
        Parameters
        ----------
        doc_id : str
            Document identifier
        language : Language
            Detected language
        confidence : float
            Detection confidence (0.0 to 1.0)
        """
        self._log_structured(
            logging.INFO,
            "language_detected",
            component="LanguageDetector",
            doc_id=doc_id,
            language=language.value if isinstance(language, Language) else str(language),
            confidence=confidence
        )
    
    def log_cache_hit(
        self,
        cache_type: str,
        key_hash: str
    ) -> None:
        """
        Log cache hit.
        
        Parameters
        ----------
        cache_type : str
            Type of cache (e.g., "embedding", "language")
        key_hash : str
            Hash of the cache key
        """
        self._log_structured(
            logging.DEBUG,
            "cache_hit",
            component="CacheManager",
            cache_type=cache_type,
            key_hash=key_hash
        )
    
    def log_fallback(
        self,
        component: str,
        reason: str,
        fallback_action: str
    ) -> None:
        """
        Log fallback activation.
        
        Parameters
        ----------
        component : str
            Component that triggered fallback
        reason : str
            Reason for fallback
        fallback_action : str
            Action taken as fallback
        """
        self._log_structured(
            logging.WARNING,
            "fallback_activated",
            component=component,
            reason=reason,
            fallback_action=fallback_action
        )
    
    def log_error(
        self,
        component: str,
        error: Exception,
        recoverable: bool,
        **context: Any
    ) -> None:
        """
        Log error with context.
        
        Parameters
        ----------
        component : str
            Component where error occurred
        error : Exception
            The exception that occurred
        recoverable : bool
            Whether the error is recoverable
        **context
            Additional context information
        """
        self._log_structured(
            logging.ERROR,
            "error_occurred",
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            recoverable=recoverable,
            **context
        )
    
    def log_timing(
        self,
        operation: str,
        duration_ms: float,
        **context: Any
    ) -> None:
        """
        Log timing information for an operation.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        duration_ms : float
            Duration in milliseconds
        **context
            Additional context information
        """
        self._log_structured(
            logging.INFO,
            "operation_timing",
            operation=operation,
            duration_ms=duration_ms,
            **context
        )

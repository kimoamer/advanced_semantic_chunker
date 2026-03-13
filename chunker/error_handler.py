"""
Error handler with graceful degradation and retry logic.

This module provides centralized error handling with:
- Automatic fallback chains for component failures
- Exponential backoff retry for network errors
- Exception wrapping with contextual information
"""

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

from chunker.exceptions import ChunkerException

logger = logging.getLogger(__name__)

__all__ = ["ErrorHandler", "CircuitBreaker", "CircuitBreakerOpenError"]


class ErrorHandler:
    """
    Handles errors with graceful degradation and retry logic.
    
    Features:
    - Fallback chains: Try primary function, fall back to secondary on error
    - Retry logic: Exponential backoff for transient errors
    - Context wrapping: Enrich exceptions with debugging information
    
    Example:
        handler = ErrorHandler(enable_fallbacks=True)
        
        # With fallback
        result = handler.with_fallback(
            primary_fn=lambda: embed_with_api(text),
            fallback_fn=lambda: use_recursive_strategy(),
            error_types=(NetworkError, APIError),
            context="embedding generation"
        )
        
        # With retry
        result = handler.with_retry(
            fn=lambda: call_api(),
            max_retries=3,
            backoff_factor=2.0,
            error_types=(NetworkError,)
        )
    """
    
    def __init__(
        self,
        enable_fallbacks: bool = True,
        log_fallbacks: bool = True,
    ):
        """
        Initialize error handler.
        
        Parameters
        ----------
        enable_fallbacks : bool
            Whether to enable automatic fallbacks (default: True)
        log_fallbacks : bool
            Whether to log fallback activations (default: True)
        """
        self.enable_fallbacks = enable_fallbacks
        self.log_fallbacks = log_fallbacks
    
    def with_fallback(
        self,
        primary_fn: Callable,
        fallback_fn: Callable,
        error_types: Tuple[Type[Exception], ...],
        context: str,
    ) -> Any:
        """
        Execute function with fallback on error.
        
        Tries to execute primary_fn. If it raises one of the specified error_types,
        executes fallback_fn instead and logs a warning.
        
        Parameters
        ----------
        primary_fn : Callable
            Primary function to execute
        fallback_fn : Callable
            Fallback function to execute on error
        error_types : Tuple[Type[Exception], ...]
            Exception types that trigger fallback
        context : str
            Description of the operation for logging
            
        Returns
        -------
        Any
            Result from primary_fn or fallback_fn
            
        Raises
        ------
        Exception
            If fallback is disabled or fallback_fn also fails
        """
        try:
            return primary_fn()
        except error_types as e:
            if not self.enable_fallbacks:
                raise
            
            if self.log_fallbacks:
                logger.warning(
                    f"Fallback activated for {context}: {type(e).__name__}: {e}. "
                    f"Using fallback strategy."
                )
            
            try:
                return fallback_fn()
            except Exception as fallback_error:
                # Fallback also failed - wrap and raise
                raise self.wrap_exception(
                    fallback_error,
                    context=f"{context} (fallback also failed)",
                    primary_error=str(e),
                )
    
    def with_retry(
        self,
        fn: Callable,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        error_types: Tuple[Type[Exception], ...] = (Exception,),
    ) -> Any:
        """
        Execute function with exponential backoff retry.
        
        Retries the function up to max_retries times with exponentially
        increasing delays between attempts.
        
        Parameters
        ----------
        fn : Callable
            Function to execute
        max_retries : int
            Maximum number of retry attempts (default: 3)
        backoff_factor : float
            Multiplier for delay between retries (default: 2.0)
        initial_delay : float
            Initial delay in seconds (default: 1.0)
        error_types : Tuple[Type[Exception], ...]
            Exception types that trigger retry (default: all exceptions)
            
        Returns
        -------
        Any
            Result from successful function execution
            
        Raises
        ------
        Exception
            If all retry attempts fail
            
        Example
        -------
        With backoff_factor=2.0 and initial_delay=1.0:
        - Attempt 1: immediate
        - Attempt 2: wait 1.0s
        - Attempt 3: wait 2.0s
        - Attempt 4: wait 4.0s
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except error_types as e:
                last_exception = e
                
                if attempt == max_retries:
                    # Final attempt failed
                    logger.error(
                        f"All {max_retries + 1} attempts failed. "
                        f"Last error: {type(e).__name__}: {e}"
                    )
                    raise
                
                # Calculate delay with exponential backoff
                delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                    f"due to {type(e).__name__}: {e}"
                )
                time.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    def wrap_exception(
        self,
        exc: Exception,
        context: str,
        **extra_info: Any,
    ) -> ChunkerException:
        """
        Wrap exception with additional context.
        
        Creates a ChunkerException that wraps the original exception
        and adds contextual information for debugging.
        
        Parameters
        ----------
        exc : Exception
            Original exception to wrap
        context : str
            Description of what was being done when error occurred
        **extra_info : Any
            Additional context information (component, operation, etc.)
            
        Returns
        -------
        ChunkerException
            Wrapped exception with context
            
        Example
        -------
        try:
            load_model()
        except Exception as e:
            raise handler.wrap_exception(
                e,
                context="loading embedding model",
                component="LazyLoadManager",
                model_name="BAAI/bge-m3"
            )
        """
        message = f"Error in {context}: {type(exc).__name__}: {exc}"

        return ChunkerException(
            message=message,
            original_exception=exc,
            context=extra_info,
        )


# ────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ────────────────────────────────────────────────────────────────────────────


class CircuitBreakerOpenError(ChunkerException):
    """Raised when a call is rejected because the circuit is open."""


class _CBState(Enum):
    CLOSED = "closed"        # Normal operation — requests pass through
    OPEN = "open"            # Too many failures — requests rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery — one trial request allowed


class CircuitBreaker:
    """
    Circuit-breaker pattern for protecting external API calls.

    State machine::

        CLOSED ──(failure_threshold failures)──► OPEN
          ▲                                        │
          └──(success in HALF_OPEN)──┐  (recovery_timeout)
                                     ▼
                                 HALF_OPEN

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before opening the circuit (default: 5).
    recovery_timeout : float
        Seconds to wait in OPEN state before moving to HALF_OPEN (default: 60).
    expected_exception : type
        Exception type (or tuple of types) that counts as a failure.
        Defaults to ``Exception`` (any exception).

    Examples
    --------
    >>> cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    >>>
    >>> def call_openai():
    ...     return embedding_provider.embed(texts)
    >>>
    >>> try:
    ...     result = cb.call(call_openai)
    ... except CircuitBreakerOpenError:
    ...     result = fallback_strategy(texts)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state = _CBState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute *fn* through the circuit breaker.

        Parameters
        ----------
        fn : Callable
            The function to protect (typically an API call).
        *args, **kwargs
            Arguments forwarded to *fn*.

        Returns
        -------
        Any
            Return value of *fn* if the circuit is CLOSED or HALF_OPEN.

        Raises
        ------
        CircuitBreakerOpenError
            If the circuit is OPEN (too many recent failures).
        Exception
            Any exception raised by *fn* (also recorded as a failure).
        """
        with self._lock:
            self._maybe_transition()

            if self._state == _CBState.OPEN:
                raise CircuitBreakerOpenError(
                    message=(
                        f"Circuit is OPEN after {self._failure_count} failures. "
                        f"Next retry in "
                        f"{self._time_until_recovery():.1f}s."
                    )
                )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise exc

    @property
    def state(self) -> str:
        """Current state as a lowercase string (``"closed"``, ``"open"``, ``"half_open"``)."""
        return self._state.value

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        with self._lock:
            self._state = _CBState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _maybe_transition(self) -> None:
        """Check if OPEN → HALF_OPEN transition is due (call inside lock)."""
        if (
            self._state == _CBState.OPEN
            and self._last_failure_time is not None
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = _CBState.HALF_OPEN
            logger.info(
                "CircuitBreaker transitioned OPEN → HALF_OPEN; "
                "allowing one trial request."
            )

    def _on_success(self) -> None:
        with self._lock:
            if self._state == _CBState.HALF_OPEN:
                logger.info("CircuitBreaker: trial request succeeded → CLOSED.")
            self._state = _CBState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == _CBState.HALF_OPEN:
                # Trial failed — go back to OPEN
                self._state = _CBState.OPEN
                logger.warning(
                    "CircuitBreaker: trial request failed → OPEN again."
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = _CBState.OPEN
                logger.error(
                    f"CircuitBreaker OPENED after {self._failure_count} "
                    f"consecutive failures."
                )

    def _time_until_recovery(self) -> float:
        """Seconds remaining before the circuit moves to HALF_OPEN."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

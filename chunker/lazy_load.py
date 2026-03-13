"""
Lazy Load Manager for the Semantic Chunker.

Manages lazy loading of expensive resources (embedding models, NLP tools)
to defer initialization until they're actually needed. Uses weak references
to allow garbage collection when resources are no longer in use.
"""

from __future__ import annotations

import logging
import sys
import threading
import weakref
from typing import Any, Dict, Optional, Union

from chunker.config import ChunkerConfig, EmbeddingProvider
from chunker.embeddings.base import BaseEmbeddingProvider
from chunker.error_handler import ErrorHandler
from chunker.exceptions import EmbeddingError, ModelLoadError
from chunker.language.arabic import ArabicProcessor
from chunker.language.english import EnglishProcessor
from chunker.models import Language

logger = logging.getLogger(__name__)


class LazyLoadManager:
    """
    Manages lazy loading of expensive resources.
    
    Defers loading of embedding models and language processors until they're
    actually needed. Uses weak references to allow garbage collection when
    resources are no longer in use. Thread-safe for concurrent access.
    
    Parameters
    ----------
    config : ChunkerConfig
        Configuration for the chunker, used to determine which resources to load.
        
    Examples
    --------
    >>> from chunker.config import ChunkerConfig, EmbeddingProvider
    >>> config = ChunkerConfig(embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER)
    >>> manager = LazyLoadManager(config)
    >>> # Embedding provider is not loaded yet
    >>> provider = manager.get_embedding_provider()  # Loads on first access
    >>> provider2 = manager.get_embedding_provider()  # Returns cached instance
    >>> assert provider is provider2
    """
    
    def __init__(self, config: ChunkerConfig):
        """Initialize lazy load manager with configuration."""
        self.config = config
        
        # Weak references to loaded resources (allow garbage collection)
        self._embedding_provider_ref: Optional[weakref.ref] = None
        self._en_processor_ref: Optional[weakref.ref] = None
        self._ar_processor_ref: Optional[weakref.ref] = None
        
        # Thread locks for concurrent access
        self._embedding_lock = threading.Lock()
        self._en_processor_lock = threading.Lock()
        self._ar_processor_lock = threading.Lock()
        
        # Memory usage tracking per component (in bytes)
        self._memory_usage: Dict[str, int] = {}
        self._memory_lock = threading.Lock()
        
        # Error handler for retry logic
        self._error_handler = ErrorHandler(
            enable_fallbacks=config.enable_fallbacks,
            log_fallbacks=True,
        )
    
    def get_embedding_provider(
        self,
        provider_type: Optional[EmbeddingProvider] = None,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> BaseEmbeddingProvider:
        """
        Get or create embedding provider on first use.
        
        Uses lazy initialization to defer loading until the embedding provider
        is actually needed. Subsequent calls return the cached instance.
        Includes retry logic for network errors during model loading.
        
        Parameters
        ----------
        provider_type : Optional[EmbeddingProvider]
            Type of embedding provider. If None, uses config.embedding_provider.
        model_name : Optional[str]
            Name of the embedding model. If None, uses config.embedding_model.
        **kwargs : Any
            Additional arguments to pass to the embedding provider constructor.
            
        Returns
        -------
        BaseEmbeddingProvider
            The embedding provider instance.
            
        Raises
        ------
        EmbeddingError
            If the provider fails to load after retries.
        ValueError
            If the provider type is unknown or unsupported.
            
        Examples
        --------
        >>> manager = LazyLoadManager(config)
        >>> provider = manager.get_embedding_provider()
        >>> embeddings = provider.embed(["Hello world"])
        """
        with self._embedding_lock:
            # Check if we have a cached instance
            if self._embedding_provider_ref is not None:
                provider = self._embedding_provider_ref()
                if provider is not None:
                    return provider
            
            # Determine provider type and model name
            provider_type = provider_type or self.config.embedding_provider
            model_name = model_name or self.config.embedding_model
            
            # Create provider with retry logic for network errors
            def create_provider():
                try:
                    if provider_type == EmbeddingProvider.SENTENCE_TRANSFORMER:
                        from chunker.embeddings.sentence_transformer import (
                            SentenceTransformerProvider,
                        )
                        return SentenceTransformerProvider(
                            model_name=model_name,
                            **kwargs
                        )
                    
                    elif provider_type == EmbeddingProvider.OPENAI:
                        from chunker.embeddings.openai_provider import OpenAIEmbeddingProvider
                        api_key = kwargs.get("api_key", self.config.openai_api_key)
                        return OpenAIEmbeddingProvider(
                            model_name=model_name,
                            api_key=api_key,
                            **{k: v for k, v in kwargs.items() if k != "api_key"}
                        )
                    
                    else:
                        raise ValueError(
                            f"Unknown embedding provider: {provider_type}. "
                            "Supported providers: SENTENCE_TRANSFORMER, OPENAI"
                        )
                except Exception as e:
                    # Wrap in EmbeddingError for proper error handling
                    raise EmbeddingError(
                        f"Failed to create embedding provider: {e}",
                        original_exception=e,
                        context={
                            "provider_type": str(provider_type),
                            "model_name": model_name,
                        }
                    )
            
            # Apply retry logic if enabled
            if self.config.retry_on_network_error:
                try:
                    provider = self._error_handler.with_retry(
                        fn=create_provider,
                        max_retries=self.config.max_retries,
                        backoff_factor=self.config.retry_backoff_factor,
                        error_types=(EmbeddingError, OSError, IOError, ConnectionError),
                    )
                except Exception as e:
                    # Wrap final failure
                    raise self._error_handler.wrap_exception(
                        e,
                        context="embedding provider creation with retry",
                        component="LazyLoadManager",
                        provider_type=str(provider_type),
                        model_name=model_name,
                    )
            else:
                provider = create_provider()
            
            # Store weak reference
            self._embedding_provider_ref = weakref.ref(provider)
            
            # Track memory usage
            self._track_memory_usage("embedding_provider", provider)
            
            return provider
    
    def get_language_processor(
        self,
        language: Language
    ) -> Union[EnglishProcessor, ArabicProcessor]:
        """
        Get or create language processor on first use.
        
        Uses lazy initialization to defer loading until the language processor
        is actually needed. Only loads the processor for the requested language.
        Subsequent calls return the cached instance.
        
        Parameters
        ----------
        language : Language
            The language for which to get the processor.
            
        Returns
        -------
        Union[EnglishProcessor, ArabicProcessor]
            The language processor instance.
            
        Raises
        ------
        ValueError
            If the language is not supported (only ENGLISH and ARABIC are supported).
            
        Examples
        --------
        >>> manager = LazyLoadManager(config)
        >>> en_proc = manager.get_language_processor(Language.ENGLISH)
        >>> sentences = en_proc.segment_sentences("Hello. World.")
        >>> ar_proc = manager.get_language_processor(Language.ARABIC)
        >>> # Only English and Arabic processors are loaded, not both initially
        """
        if language == Language.ENGLISH:
            with self._en_processor_lock:
                # Check if we have a cached instance
                if self._en_processor_ref is not None:
                    processor = self._en_processor_ref()
                    if processor is not None:
                        return processor
                
                # Create new processor instance
                processor = EnglishProcessor()
                
                # Store weak reference
                self._en_processor_ref = weakref.ref(processor)
                
                # Track memory usage
                self._track_memory_usage("english_processor", processor)
                
                return processor
        
        elif language == Language.ARABIC:
            with self._ar_processor_lock:
                # Check if we have a cached instance
                if self._ar_processor_ref is not None:
                    processor = self._ar_processor_ref()
                    if processor is not None:
                        return processor
                
                # Create new processor instance
                processor = ArabicProcessor(
                    normalize_alef=self.config.arabic_normalize_alef,
                    normalize_yeh=self.config.arabic_normalize_yeh,
                    normalize_teh_marbuta=self.config.arabic_normalize_teh_marbuta,
                    remove_tashkeel=self.config.arabic_remove_tashkeel,
                    remove_tatweel=self.config.arabic_remove_tatweel,
                    normalize_punctuation=self.config.arabic_normalize_punctuation,
                )
                
                # Store weak reference
                self._ar_processor_ref = weakref.ref(processor)
                
                # Track memory usage
                self._track_memory_usage("arabic_processor", processor)
                
                return processor
        
        else:
            raise ValueError(
                f"Unsupported language: {language}. "
                "Only ENGLISH and ARABIC are supported."
            )
    
    def _track_memory_usage(self, component: str, obj: Any) -> None:
        """
        Track memory usage for a component.
        
        Estimates the memory usage of an object using sys.getsizeof.
        This is an approximation and may not capture all memory used by
        complex objects with nested structures.
        
        Parameters
        ----------
        component : str
            Name of the component (e.g., "embedding_provider", "english_processor").
        obj : Any
            The object to measure.
        """
        with self._memory_lock:
            try:
                # Get approximate size in bytes
                size = sys.getsizeof(obj)
                
                # For objects with __dict__, also count attributes
                if hasattr(obj, '__dict__'):
                    for attr_value in obj.__dict__.values():
                        try:
                            size += sys.getsizeof(attr_value)
                        except (TypeError, AttributeError):
                            # Skip objects that can't be sized
                            pass
                
                self._memory_usage[component] = size
                logger.debug(f"Tracked memory usage for {component}: {size:,} bytes")
            except Exception as e:
                # Don't fail if memory tracking fails
                logger.warning(f"Failed to track memory usage for {component}: {e}")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for loaded components.
        
        Returns a dictionary mapping component names to their estimated
        memory usage in bytes. Only includes components that have been loaded.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping component names to memory usage in bytes.
            Possible keys: "embedding_provider", "english_processor", "arabic_processor".
            
        Examples
        --------
        >>> manager = LazyLoadManager(config)
        >>> manager.preload(["embedding_provider", "english_processor"])
        >>> usage = manager.get_memory_usage()
        >>> print(f"Embedding provider: {usage['embedding_provider']:,} bytes")
        >>> print(f"Total: {sum(usage.values()):,} bytes")
        """
        with self._memory_lock:
            return self._memory_usage.copy()
    
    def preload(self, components: list[str]) -> None:
        """
        Explicitly preload components for predictable latency.
        
        Useful for production warmup to avoid cold start latency on first request.
        
        Parameters
        ----------
        components : list[str]
            List of component names to preload. Valid values:
            - "embedding_provider": Load the embedding provider
            - "english_processor": Load the English language processor
            - "arabic_processor": Load the Arabic language processor
            
        Examples
        --------
        >>> manager = LazyLoadManager(config)
        >>> # Preload components during application startup
        >>> manager.preload(["embedding_provider", "english_processor"])
        >>> # Now first requests will be faster
        """
        for component in components:
            if component == "embedding_provider":
                self.get_embedding_provider()
            elif component == "english_processor":
                self.get_language_processor(Language.ENGLISH)
            elif component == "arabic_processor":
                self.get_language_processor(Language.ARABIC)
            else:
                raise ValueError(
                    f"Unknown component: {component}. "
                    "Valid components: embedding_provider, english_processor, arabic_processor"
                )
    
    def clear(self) -> None:
        """
        Clear all loaded resources to free memory.
        
        Removes all weak references, allowing garbage collection to reclaim
        memory used by loaded models and processors. Subsequent calls to
        get_embedding_provider() or get_language_processor() will reload
        the resources. Also clears memory usage tracking.
        
        Examples
        --------
        >>> manager = LazyLoadManager(config)
        >>> provider = manager.get_embedding_provider()
        >>> manager.clear()  # Free memory
        >>> # Next call will reload the provider
        >>> provider2 = manager.get_embedding_provider()
        """
        with self._embedding_lock:
            self._embedding_provider_ref = None
        
        with self._en_processor_lock:
            self._en_processor_ref = None
        
        with self._ar_processor_lock:
            self._ar_processor_ref = None
        
        with self._memory_lock:
            self._memory_usage.clear()

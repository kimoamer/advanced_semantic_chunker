"""
Tests for error handler and exception hierarchy.

Tests the ErrorHandler class and its integration with SemanticChunker,
including fallback chains, retry logic, and exception wrapping.
"""

import pytest
from unittest.mock import Mock, patch

from chunker.config import ChunkerConfig, StrategyType
from chunker.core import SemanticChunker
from chunker.error_handler import ErrorHandler
from chunker.exceptions import (
    ChunkerException,
    ConfigurationError,
    EmbeddingError,
    InputValidationError,
    LanguageDetectionError,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy and context information."""
    
    def test_chunker_exception_with_context(self):
        """Test ChunkerException includes context information."""
        exc = ChunkerException(
            "Test error",
            context={"component": "test", "operation": "test_op"}
        )
        
        assert "Test error" in str(exc)
        assert exc.context["component"] == "test"
        assert exc.context["operation"] == "test_op"
    
    def test_chunker_exception_with_original(self):
        """Test ChunkerException wraps original exception."""
        original = ValueError("Original error")
        exc = ChunkerException(
            "Wrapped error",
            original_exception=original,
            context={"component": "test"}
        )
        
        assert exc.original_exception is original
        assert "ValueError" in str(exc)
        assert "Original error" in str(exc)
    
    def test_exception_subclasses(self):
        """Test exception hierarchy is correct."""
        assert issubclass(ConfigurationError, ChunkerException)
        assert issubclass(InputValidationError, ChunkerException)
        assert issubclass(EmbeddingError, ChunkerException)
        assert issubclass(LanguageDetectionError, ChunkerException)


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    def test_with_fallback_success(self):
        """Test fallback is not used when primary succeeds."""
        handler = ErrorHandler(enable_fallbacks=True)
        
        primary_called = False
        fallback_called = False
        
        def primary():
            nonlocal primary_called
            primary_called = True
            return "primary result"
        
        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback result"
        
        result = handler.with_fallback(
            primary_fn=primary,
            fallback_fn=fallback,
            error_types=(ValueError,),
            context="test operation"
        )
        
        assert result == "primary result"
        assert primary_called
        assert not fallback_called
    
    def test_with_fallback_on_error(self):
        """Test fallback is used when primary fails."""
        handler = ErrorHandler(enable_fallbacks=True)
        
        def primary():
            raise ValueError("Primary failed")
        
        def fallback():
            return "fallback result"
        
        result = handler.with_fallback(
            primary_fn=primary,
            fallback_fn=fallback,
            error_types=(ValueError,),
            context="test operation"
        )
        
        assert result == "fallback result"
    
    def test_with_fallback_disabled(self):
        """Test fallback is not used when disabled."""
        handler = ErrorHandler(enable_fallbacks=False)
        
        def primary():
            raise ValueError("Primary failed")
        
        def fallback():
            return "fallback result"
        
        with pytest.raises(ValueError, match="Primary failed"):
            handler.with_fallback(
                primary_fn=primary,
                fallback_fn=fallback,
                error_types=(ValueError,),
                context="test operation"
            )
    
    def test_with_retry_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        handler = ErrorHandler()
        
        call_count = 0
        
        def fn():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = handler.with_retry(fn, max_retries=3)
        
        assert result == "success"
        assert call_count == 1
    
    def test_with_retry_success_after_failures(self):
        """Test retry succeeds after some failures."""
        handler = ErrorHandler()
        
        call_count = 0
        
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = handler.with_retry(
            fn,
            max_retries=3,
            backoff_factor=1.0,
            initial_delay=0.01,  # Fast for testing
            error_types=(ConnectionError,)
        )
        
        assert result == "success"
        assert call_count == 3
    
    def test_with_retry_all_attempts_fail(self):
        """Test retry raises after all attempts fail."""
        handler = ErrorHandler()
        
        call_count = 0
        
        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError, match="Network error"):
            handler.with_retry(
                fn,
                max_retries=2,
                backoff_factor=1.0,
                initial_delay=0.01,
                error_types=(ConnectionError,)
            )
        
        assert call_count == 3  # Initial + 2 retries
    
    def test_wrap_exception(self):
        """Test exception wrapping with context."""
        handler = ErrorHandler()
        
        original = ValueError("Original error")
        wrapped = handler.wrap_exception(
            original,
            context="test operation",
            component="TestComponent",
            extra_info="test"
        )
        
        assert isinstance(wrapped, ChunkerException)
        assert wrapped.original_exception is original
        assert wrapped.context["component"] == "TestComponent"
        assert wrapped.context["extra_info"] == "test"


class TestSemanticChunkerErrorHandling:
    """Test error handling integration in SemanticChunker."""
    
    def test_input_validation_none(self):
        """Test that None input raises InputValidationError."""
        config = ChunkerConfig(strategy=StrategyType.SENTENCE)
        chunker = SemanticChunker(config)
        
        with pytest.raises(InputValidationError, match="cannot be None"):
            chunker.chunk(None)
    
    def test_input_validation_wrong_type(self):
        """Test that non-string input raises InputValidationError."""
        config = ChunkerConfig(strategy=StrategyType.SENTENCE)
        chunker = SemanticChunker(config)
        
        with pytest.raises(InputValidationError, match="must be a string"):
            chunker.chunk(123)
    
    def test_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list without error."""
        config = ChunkerConfig(strategy=StrategyType.SENTENCE)
        chunker = SemanticChunker(config)
        
        result = chunker.chunk("")
        assert result == []
        
        result = chunker.chunk("   ")
        assert result == []
    
    def test_language_detection_fallback(self):
        """Test that language detection falls back to English on failure."""
        config = ChunkerConfig(
            strategy=StrategyType.SENTENCE,
            enable_fallbacks=True,
        )
        chunker = SemanticChunker(config)
        
        # Mock the detector to raise an exception
        with patch.object(chunker._detector, 'detect', side_effect=Exception("Detection failed")):
            # Should not raise, should fall back to English
            chunks = chunker.chunk("This is a test.")
            
            # Should still produce chunks
            assert len(chunks) > 0
    
    def test_embedding_provider_fallback_to_recursive(self):
        """Test that embedding provider failure falls back to recursive strategy."""
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            enable_fallbacks=True,
        )
        chunker = SemanticChunker(config)
        
        # Mock the lazy load manager to raise an exception when getting embedding provider
        with patch.object(
            chunker._lazy_load_manager,
            'get_embedding_provider',
            side_effect=EmbeddingError("Provider failed")
        ):
            # Should not raise, should fall back to recursive strategy
            chunks = chunker.chunk("This is a test. This is another sentence.")
            
            # Should still produce chunks using recursive strategy
            assert len(chunks) > 0
    
    def test_exception_wrapping_with_context(self):
        """Test that unexpected exceptions are wrapped with context."""
        config = ChunkerConfig(strategy=StrategyType.SENTENCE)
        chunker = SemanticChunker(config)
        
        # Mock normalizer to raise an unexpected exception
        with patch.object(
            chunker._normalizer,
            'normalize',
            side_effect=RuntimeError("Unexpected error")
        ):
            with pytest.raises(ChunkerException) as exc_info:
                chunker.chunk("Test text")
            
            # Check that exception is wrapped with context
            exc = exc_info.value
            assert exc.original_exception is not None
            assert isinstance(exc.original_exception, RuntimeError)
            assert "document chunking" in str(exc)


class TestRetryLogicIntegration:
    """Test retry logic integration in LazyLoadManager."""
    
    def test_embedding_provider_retry_on_network_error(self):
        """Test that embedding provider creation retries on network errors."""
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            retry_on_network_error=True,
            max_retries=2,
            retry_backoff_factor=1.0,
        )
        chunker = SemanticChunker(config)
        
        call_count = 0
        
        def mock_provider_init(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            # Return a mock provider on success
            mock = Mock()
            mock.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
            mock.get_dimension = Mock(return_value=3)
            return mock
        
        # Patch the provider class to simulate network errors
        with patch(
            'chunker.embeddings.sentence_transformer.SentenceTransformerProvider',
            side_effect=mock_provider_init
        ):
            # Should succeed after retry
            provider = chunker._lazy_load_manager.get_embedding_provider()
            assert provider is not None
            assert call_count == 2  # Failed once, succeeded on retry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

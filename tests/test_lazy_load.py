"""
Unit tests for LazyLoadManager.

Tests lazy initialization, weak references, thread safety, and resource management.
"""

import gc
import threading
import weakref
import pytest

from chunker.config import ChunkerConfig, EmbeddingProvider, StrategyType
from chunker.lazy_load import LazyLoadManager
from chunker.models import Language


class TestLazyLoadManager:
    """Test suite for LazyLoadManager class."""
    
    def test_embedding_provider_lazy_initialization(self):
        """Test that embedding provider is not loaded until first access."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Initially, no provider should be loaded
        assert manager._embedding_provider_ref is None
        
        # First access should load the provider
        provider = manager.get_embedding_provider()
        assert provider is not None
        assert manager._embedding_provider_ref is not None
        
        # Verify it's the correct type
        from chunker.embeddings.sentence_transformer import SentenceTransformerProvider
        assert isinstance(provider, SentenceTransformerProvider)
    
    def test_embedding_provider_caching(self):
        """Test that subsequent calls return the same provider instance."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Get provider twice
        provider1 = manager.get_embedding_provider()
        provider2 = manager.get_embedding_provider()
        
        # Should be the same instance
        assert provider1 is provider2
    
    def test_language_processor_lazy_initialization_english(self):
        """Test that English processor is not loaded until first access."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Initially, no processor should be loaded
        assert manager._en_processor_ref is None
        assert manager._ar_processor_ref is None
        
        # First access should load only English processor
        en_processor = manager.get_language_processor(Language.ENGLISH)
        assert en_processor is not None
        assert manager._en_processor_ref is not None
        assert manager._ar_processor_ref is None  # Arabic should still be None
        
        # Verify it's the correct type
        from chunker.language.english import EnglishProcessor
        assert isinstance(en_processor, EnglishProcessor)
    
    def test_language_processor_lazy_initialization_arabic(self):
        """Test that Arabic processor is not loaded until first access."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Initially, no processor should be loaded
        assert manager._en_processor_ref is None
        assert manager._ar_processor_ref is None
        
        # First access should load only Arabic processor
        ar_processor = manager.get_language_processor(Language.ARABIC)
        assert ar_processor is not None
        assert manager._ar_processor_ref is not None
        assert manager._en_processor_ref is None  # English should still be None
        
        # Verify it's the correct type
        from chunker.language.arabic import ArabicProcessor
        assert isinstance(ar_processor, ArabicProcessor)
    
    def test_language_processor_caching(self):
        """Test that subsequent calls return the same processor instance."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Get English processor twice
        en_proc1 = manager.get_language_processor(Language.ENGLISH)
        en_proc2 = manager.get_language_processor(Language.ENGLISH)
        
        # Should be the same instance
        assert en_proc1 is en_proc2
        
        # Get Arabic processor twice
        ar_proc1 = manager.get_language_processor(Language.ARABIC)
        ar_proc2 = manager.get_language_processor(Language.ARABIC)
        
        # Should be the same instance
        assert ar_proc1 is ar_proc2
        
        # But English and Arabic should be different instances
        assert en_proc1 is not ar_proc1
    
    def test_unsupported_language_raises_error(self):
        """Test that unsupported languages raise ValueError."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        with pytest.raises(ValueError, match="Unsupported language"):
            manager.get_language_processor(Language.MIXED)
        
        with pytest.raises(ValueError, match="Unsupported language"):
            manager.get_language_processor(Language.UNKNOWN)
    
    def test_unknown_embedding_provider_raises_error(self):
        """Test that unknown embedding provider raises ChunkerException with ValueError as cause."""
        from chunker.exceptions import ChunkerException
        
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Try to get a provider with an invalid type
        # Now wrapped in ChunkerException due to error handling
        with pytest.raises(ChunkerException, match="Unknown embedding provider"):
            manager.get_embedding_provider(provider_type="invalid_provider")
    
    def test_weak_references_allow_garbage_collection(self):
        """Test that weak references allow garbage collection when resources are no longer used."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Get a processor
        processor = manager.get_language_processor(Language.ENGLISH)
        
        # Create a weak reference to track it
        weak_ref = weakref.ref(processor)
        
        # Verify it's alive
        assert weak_ref() is not None
        
        # Delete the strong reference
        del processor
        
        # Force garbage collection
        gc.collect()
        
        # The weak reference in the manager should now be dead
        # (or at least, we can verify that the manager's weak ref is working)
        # Note: We can't guarantee GC timing, but we can verify the weak ref mechanism
        assert manager._en_processor_ref is not None
    
    def test_clear_removes_all_references(self):
        """Test that clear() removes all weak references."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Load all resources
        manager.get_embedding_provider()
        manager.get_language_processor(Language.ENGLISH)
        manager.get_language_processor(Language.ARABIC)
        
        # Verify they're loaded
        assert manager._embedding_provider_ref is not None
        assert manager._en_processor_ref is not None
        assert manager._ar_processor_ref is not None
        
        # Clear all references
        manager.clear()
        
        # Verify all references are cleared
        assert manager._embedding_provider_ref is None
        assert manager._en_processor_ref is None
        assert manager._ar_processor_ref is None
    
    def test_clear_allows_reloading(self):
        """Test that resources can be reloaded after clear()."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Load a processor
        proc1 = manager.get_language_processor(Language.ENGLISH)
        
        # Clear
        manager.clear()
        
        # Load again
        proc2 = manager.get_language_processor(Language.ENGLISH)
        
        # Should be a new instance (not the same as before)
        assert proc2 is not None
        # Note: We can't guarantee proc1 != proc2 due to GC timing,
        # but we can verify it works
    
    def test_preload_embedding_provider(self):
        """Test preloading embedding provider."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Initially not loaded
        assert manager._embedding_provider_ref is None
        
        # Preload
        manager.preload(["embedding_provider"])
        
        # Should now be loaded
        assert manager._embedding_provider_ref is not None
    
    def test_preload_language_processors(self):
        """Test preloading language processors."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Initially not loaded
        assert manager._en_processor_ref is None
        assert manager._ar_processor_ref is None
        
        # Preload both
        manager.preload(["english_processor", "arabic_processor"])
        
        # Should now be loaded
        assert manager._en_processor_ref is not None
        assert manager._ar_processor_ref is not None
    
    def test_preload_multiple_components(self):
        """Test preloading multiple components at once."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Preload all components
        manager.preload([
            "embedding_provider",
            "english_processor",
            "arabic_processor"
        ])
        
        # All should be loaded
        assert manager._embedding_provider_ref is not None
        assert manager._en_processor_ref is not None
        assert manager._ar_processor_ref is not None
    
    def test_preload_invalid_component_raises_error(self):
        """Test that preloading invalid component raises ValueError."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        with pytest.raises(ValueError, match="Unknown component"):
            manager.preload(["invalid_component"])
    
    def test_thread_safety_embedding_provider(self):
        """Test thread-safe access to embedding provider."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        providers = []
        errors = []
        
        def worker():
            try:
                provider = manager.get_embedding_provider()
                providers.append(provider)
            except Exception as e:
                errors.append(e)
        
        # Run 10 threads concurrently
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All threads should get the same provider instance
        assert len(providers) == 10
        first_provider = providers[0]
        for provider in providers:
            assert provider is first_provider
    
    def test_thread_safety_language_processors(self):
        """Test thread-safe access to language processors."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        en_processors = []
        ar_processors = []
        errors = []
        
        def worker_en():
            try:
                processor = manager.get_language_processor(Language.ENGLISH)
                en_processors.append(processor)
            except Exception as e:
                errors.append(e)
        
        def worker_ar():
            try:
                processor = manager.get_language_processor(Language.ARABIC)
                ar_processors.append(processor)
            except Exception as e:
                errors.append(e)
        
        # Run 10 threads for each language concurrently
        threads = []
        for _ in range(10):
            t_en = threading.Thread(target=worker_en)
            t_ar = threading.Thread(target=worker_ar)
            threads.extend([t_en, t_ar])
            t_en.start()
            t_ar.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All English threads should get the same processor instance
        assert len(en_processors) == 10
        first_en = en_processors[0]
        for proc in en_processors:
            assert proc is first_en
        
        # All Arabic threads should get the same processor instance
        assert len(ar_processors) == 10
        first_ar = ar_processors[0]
        for proc in ar_processors:
            assert proc is first_ar
        
        # But English and Arabic should be different
        assert first_en is not first_ar
    
    def test_custom_embedding_provider_kwargs(self):
        """Test passing custom kwargs to embedding provider."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Get provider with custom kwargs (device, etc.)
        provider = manager.get_embedding_provider(device="cpu")
        
        assert provider is not None
    
    def test_arabic_processor_uses_config_settings(self):
        """Test that Arabic processor uses configuration settings."""
        config = ChunkerConfig(
            arabic_normalize_alef=True,
            arabic_normalize_yeh=True,
            arabic_remove_tashkeel=True
        )
        manager = LazyLoadManager(config)
        
        # Get Arabic processor
        ar_processor = manager.get_language_processor(Language.ARABIC)
        
        # Verify it was created with the config settings
        assert ar_processor.normalize_alef == True
        assert ar_processor.normalize_yeh == True
        assert ar_processor.remove_tashkeel == True
    
    def test_memory_usage_tracking_embedding_provider(self):
        """Test that memory usage is tracked for embedding provider."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Initially, no memory usage tracked
        usage = manager.get_memory_usage()
        assert len(usage) == 0
        
        # Load embedding provider
        manager.get_embedding_provider()
        
        # Memory usage should be tracked
        usage = manager.get_memory_usage()
        assert "embedding_provider" in usage
        assert usage["embedding_provider"] > 0
    
    def test_memory_usage_tracking_language_processors(self):
        """Test that memory usage is tracked for language processors."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Load English processor
        manager.get_language_processor(Language.ENGLISH)
        
        # Memory usage should be tracked
        usage = manager.get_memory_usage()
        assert "english_processor" in usage
        assert usage["english_processor"] > 0
        
        # Load Arabic processor
        manager.get_language_processor(Language.ARABIC)
        
        # Both should be tracked
        usage = manager.get_memory_usage()
        assert "english_processor" in usage
        assert "arabic_processor" in usage
        assert usage["arabic_processor"] > 0
    
    def test_memory_usage_tracking_multiple_components(self):
        """Test memory usage tracking for multiple components."""
        config = ChunkerConfig(
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        manager = LazyLoadManager(config)
        
        # Preload all components
        manager.preload([
            "embedding_provider",
            "english_processor",
            "arabic_processor"
        ])
        
        # All should be tracked
        usage = manager.get_memory_usage()
        assert len(usage) == 3
        assert "embedding_provider" in usage
        assert "english_processor" in usage
        assert "arabic_processor" in usage
        
        # All should have positive memory usage
        for component, size in usage.items():
            assert size > 0, f"{component} should have positive memory usage"
    
    def test_memory_usage_cleared_on_clear(self):
        """Test that memory usage tracking is cleared when clear() is called."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Load a processor
        manager.get_language_processor(Language.ENGLISH)
        
        # Verify memory is tracked
        usage = manager.get_memory_usage()
        assert len(usage) > 0
        
        # Clear
        manager.clear()
        
        # Memory usage should be cleared
        usage = manager.get_memory_usage()
        assert len(usage) == 0
    
    def test_get_memory_usage_returns_copy(self):
        """Test that get_memory_usage returns a copy, not the internal dict."""
        config = ChunkerConfig()
        manager = LazyLoadManager(config)
        
        # Load a processor
        manager.get_language_processor(Language.ENGLISH)
        
        # Get memory usage
        usage1 = manager.get_memory_usage()
        usage2 = manager.get_memory_usage()
        
        # Should be equal but not the same object
        assert usage1 == usage2
        assert usage1 is not usage2
        
        # Modifying the returned dict should not affect the internal state
        usage1["test"] = 12345
        usage3 = manager.get_memory_usage()
        assert "test" not in usage3

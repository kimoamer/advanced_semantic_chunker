"""
Unit tests for CacheManager.

Tests basic functionality, LRU eviction, thread safety, and statistics tracking.
"""

import threading
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from chunker.cache import CacheManager
from chunker.models import Language


class TestCacheManager:
    """Test suite for CacheManager class."""
    
    def test_embedding_cache_basic(self):
        """Test basic embedding cache get/set operations."""
        cache = CacheManager(embedding_cache_size=100)
        
        # Create test embedding
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        text = "Hello world"
        model = "test-model"
        
        # Initially should be cache miss
        result = cache.get_embedding(text, model)
        assert result is None
        
        # Set embedding
        cache.set_embedding(text, model, embedding)
        
        # Should now be cache hit
        result = cache.get_embedding(text, model)
        assert result is not None
        assert np.array_equal(result, embedding)
    
    def test_embedding_cache_different_models(self):
        """Test that same text with different models creates different cache entries."""
        cache = CacheManager()
        
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        text = "Same text"
        
        cache.set_embedding(text, "model-1", embedding1)
        cache.set_embedding(text, "model-2", embedding2)
        
        # Should retrieve different embeddings for different models
        result1 = cache.get_embedding(text, "model-1")
        result2 = cache.get_embedding(text, "model-2")
        
        assert np.array_equal(result1, embedding1)
        assert np.array_equal(result2, embedding2)
    
    def test_embedding_lru_eviction(self):
        """Test LRU eviction when cache size is exceeded."""
        cache = CacheManager(embedding_cache_size=3)
        
        # Add 3 embeddings (fill cache)
        for i in range(3):
            embedding = np.array([float(i)])
            cache.set_embedding(f"text-{i}", "model", embedding)
        
        # Verify all 3 are cached
        for i in range(3):
            result = cache.get_embedding(f"text-{i}", "model")
            assert result is not None
        
        # Add 4th embedding (should evict text-0, the LRU)
        cache.set_embedding("text-3", "model", np.array([3.0]))
        
        # text-0 should be evicted
        result = cache.get_embedding("text-0", "model")
        assert result is None
        
        # text-1, text-2, text-3 should still be cached
        for i in range(1, 4):
            result = cache.get_embedding(f"text-{i}", "model")
            assert result is not None
    
    def test_embedding_lru_access_updates_order(self):
        """Test that accessing an entry moves it to most recently used."""
        cache = CacheManager(embedding_cache_size=3)
        
        # Add 3 embeddings
        for i in range(3):
            cache.set_embedding(f"text-{i}", "model", np.array([float(i)]))
        
        # Access text-0 (moves it to end)
        cache.get_embedding("text-0", "model")
        
        # Add text-3 (should evict text-1, not text-0)
        cache.set_embedding("text-3", "model", np.array([3.0]))
        
        # text-0 should still be cached (was accessed recently)
        assert cache.get_embedding("text-0", "model") is not None
        
        # text-1 should be evicted (was LRU)
        assert cache.get_embedding("text-1", "model") is None
        
        # text-2 and text-3 should be cached
        assert cache.get_embedding("text-2", "model") is not None
        assert cache.get_embedding("text-3", "model") is not None
    
    def test_language_cache_basic(self):
        """Test basic language cache get/set operations."""
        cache = CacheManager(lang_detect_cache_size=100)
        
        text = "Hello world"
        language = Language.ENGLISH
        
        # Initially should be cache miss
        result = cache.get_language(text)
        assert result is None
        
        # Set language
        cache.set_language(text, language)
        
        # Should now be cache hit
        result = cache.get_language(text)
        assert result == language
    
    def test_language_lru_eviction(self):
        """Test LRU eviction for language cache."""
        cache = CacheManager(lang_detect_cache_size=2)
        
        # Add 2 language detections (fill cache)
        cache.set_language("text-0", Language.ENGLISH)
        cache.set_language("text-1", Language.ARABIC)
        
        # Verify both are cached
        assert cache.get_language("text-0") == Language.ENGLISH
        assert cache.get_language("text-1") == Language.ARABIC
        
        # Add 3rd language detection (should evict text-0)
        cache.set_language("text-2", Language.MIXED)
        
        # text-0 should be evicted
        assert cache.get_language("text-0") is None
        
        # text-1 and text-2 should still be cached
        assert cache.get_language("text-1") == Language.ARABIC
        assert cache.get_language("text-2") == Language.MIXED
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = CacheManager(embedding_cache_size=10, lang_detect_cache_size=5)
        
        # Initial stats should be zero
        stats = cache.get_stats()
        assert stats["embedding_hits"] == 0
        assert stats["embedding_misses"] == 0
        assert stats["embedding_evictions"] == 0
        assert stats["embedding_total_accesses"] == 0
        assert stats["language_hits"] == 0
        assert stats["language_misses"] == 0
        assert stats["language_evictions"] == 0
        assert stats["language_total_accesses"] == 0
        
        # Add embedding and access it
        embedding = np.array([0.1, 0.2])
        cache.set_embedding("text", "model", embedding)
        cache.get_embedding("text", "model")  # Hit
        cache.get_embedding("other", "model")  # Miss
        
        # Add language and access it
        cache.set_language("text", Language.ENGLISH)
        cache.get_language("text")  # Hit
        cache.get_language("other")  # Miss
        
        # Check stats
        stats = cache.get_stats()
        assert stats["embedding_hits"] == 1
        assert stats["embedding_misses"] == 1
        assert stats["embedding_hit_rate"] == 0.5
        assert stats["embedding_total_accesses"] == 2
        assert stats["embedding_evictions"] == 0
        assert stats["language_hits"] == 1
        assert stats["language_misses"] == 1
        assert stats["language_hit_rate"] == 0.5
        assert stats["language_total_accesses"] == 2
        assert stats["language_evictions"] == 0
        assert stats["embedding_cache_size"] == 1
        assert stats["language_cache_size"] == 1
    
    def test_eviction_statistics(self):
        """Test that eviction counts are tracked correctly."""
        cache = CacheManager(embedding_cache_size=3, lang_detect_cache_size=2)
        
        # Fill embedding cache
        for i in range(3):
            cache.set_embedding(f"text-{i}", "model", np.array([float(i)]))
        
        # Verify no evictions yet
        stats = cache.get_stats()
        assert stats["embedding_evictions"] == 0
        
        # Add one more to trigger eviction
        cache.set_embedding("text-3", "model", np.array([3.0]))
        
        # Verify eviction count increased
        stats = cache.get_stats()
        assert stats["embedding_evictions"] == 1
        
        # Add two more to trigger two more evictions
        cache.set_embedding("text-4", "model", np.array([4.0]))
        cache.set_embedding("text-5", "model", np.array([5.0]))
        
        # Verify eviction count is now 3
        stats = cache.get_stats()
        assert stats["embedding_evictions"] == 3
        
        # Test language cache evictions
        cache.set_language("lang-0", Language.ENGLISH)
        cache.set_language("lang-1", Language.ARABIC)
        
        # No evictions yet
        stats = cache.get_stats()
        assert stats["language_evictions"] == 0
        
        # Trigger eviction
        cache.set_language("lang-2", Language.MIXED)
        
        # Verify eviction count increased
        stats = cache.get_stats()
        assert stats["language_evictions"] == 1
    
    def test_clear_cache(self):
        """Test clearing all caches."""
        cache = CacheManager()
        
        # Add some data
        cache.set_embedding("text", "model", np.array([0.1]))
        cache.set_language("text", Language.ENGLISH)
        
        # Verify data is cached
        assert cache.get_embedding("text", "model") is not None
        assert cache.get_language("text") is not None
        
        # Clear cache
        cache.clear()
        
        # Verify stats are reset immediately after clear
        stats = cache.get_stats()
        assert stats["embedding_hits"] == 0
        assert stats["embedding_misses"] == 0
        assert stats["language_hits"] == 0
        assert stats["language_misses"] == 0
        assert stats["embedding_cache_size"] == 0
        assert stats["language_cache_size"] == 0
        
        # Verify data is gone (this will increment miss counters)
        assert cache.get_embedding("text", "model") is None
        assert cache.get_language("text") is None
    
    def test_thread_safety_embedding_cache(self):
        """Test thread-safe access to embedding cache."""
        cache = CacheManager(embedding_cache_size=1000)
        errors = []
        
        def worker(thread_id: int):
            try:
                for i in range(100):
                    text = f"thread-{thread_id}-text-{i}"
                    embedding = np.array([float(thread_id), float(i)])
                    cache.set_embedding(text, "model", embedding)
                    result = cache.get_embedding(text, "model")
                    assert result is not None
                    assert np.array_equal(result, embedding)
            except Exception as e:
                errors.append(e)
        
        # Run 10 threads concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_thread_safety_language_cache(self):
        """Test thread-safe access to language cache."""
        cache = CacheManager(lang_detect_cache_size=1000)
        errors = []
        languages = [Language.ENGLISH, Language.ARABIC, Language.MIXED]
        
        def worker(thread_id: int):
            try:
                for i in range(100):
                    text = f"thread-{thread_id}-text-{i}"
                    language = languages[i % len(languages)]
                    cache.set_language(text, language)
                    result = cache.get_language(text)
                    assert result == language
            except Exception as e:
                errors.append(e)
        
        # Run 10 threads concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_disk_persistence_requires_cache_dir(self):
        """Test that disk persistence methods require cache_dir to be configured."""
        cache = CacheManager()
        
        with pytest.raises(ValueError, match="cache_dir must be configured"):
            cache.save_to_disk()
        
        with pytest.raises(ValueError, match="cache_dir must be configured"):
            cache.load_from_disk()
    
    def test_save_and_load_embedding_cache(self, tmp_path):
        """Test saving and loading embedding cache to/from disk."""
        cache_dir = str(tmp_path / "cache")
        
        # Create cache and add some embeddings
        cache1 = CacheManager(cache_dir=cache_dir)
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        
        cache1.set_embedding("text1", "model1", embedding1)
        cache1.set_embedding("text2", "model2", embedding2)
        
        # Access one to create hit statistics
        cache1.get_embedding("text1", "model1")
        
        # Get stats before save
        stats_before_save = cache1.get_stats()
        
        # Save to disk
        cache1.save_to_disk()
        
        # Create new cache instance and load from disk
        cache2 = CacheManager(cache_dir=cache_dir)
        cache2.load_from_disk()
        
        # Verify statistics are restored immediately after load
        stats_after_load = cache2.get_stats()
        assert stats_after_load["embedding_hits"] == stats_before_save["embedding_hits"]
        assert stats_after_load["embedding_misses"] == stats_before_save["embedding_misses"]
        
        # Verify embeddings are restored
        result1 = cache2.get_embedding("text1", "model1")
        result2 = cache2.get_embedding("text2", "model2")
        
        assert result1 is not None
        assert result2 is not None
        assert np.array_equal(result1, embedding1)
        assert np.array_equal(result2, embedding2)
    
    def test_save_and_load_language_cache(self, tmp_path):
        """Test saving and loading language cache to/from disk."""
        cache_dir = str(tmp_path / "cache")
        
        # Create cache and add some language detections
        cache1 = CacheManager(cache_dir=cache_dir)
        cache1.set_language("Hello world", Language.ENGLISH)
        cache1.set_language("مرحبا", Language.ARABIC)
        
        # Access one to create hit statistics
        cache1.get_language("Hello world")
        
        # Get stats before save
        stats_before_save = cache1.get_stats()
        
        # Save to disk
        cache1.save_to_disk()
        
        # Create new cache instance and load from disk
        cache2 = CacheManager(cache_dir=cache_dir)
        cache2.load_from_disk()
        
        # Verify statistics are restored immediately after load
        stats_after_load = cache2.get_stats()
        assert stats_after_load["language_hits"] == stats_before_save["language_hits"]
        assert stats_after_load["language_misses"] == stats_before_save["language_misses"]
        
        # Verify languages are restored
        result1 = cache2.get_language("Hello world")
        result2 = cache2.get_language("مرحبا")
        
        assert result1 == Language.ENGLISH
        assert result2 == Language.ARABIC
    
    def test_load_from_nonexistent_cache(self, tmp_path):
        """Test loading from disk when cache files don't exist."""
        cache_dir = str(tmp_path / "nonexistent")
        
        # Should not raise an error, just initialize with empty cache
        cache = CacheManager(cache_dir=cache_dir)
        cache.load_from_disk()
        
        # Verify cache is empty
        stats = cache.get_stats()
        assert stats["embedding_cache_size"] == 0
        assert stats["language_cache_size"] == 0
    
    def test_save_creates_cache_directory(self, tmp_path):
        """Test that save_to_disk creates the cache directory if it doesn't exist."""
        cache_dir = str(tmp_path / "new_cache_dir")
        
        cache = CacheManager(cache_dir=cache_dir)
        cache.set_embedding("text", "model", np.array([0.1]))
        
        # Directory shouldn't exist yet
        import os
        assert not os.path.exists(cache_dir)
        
        # Save should create it
        cache.save_to_disk()
        assert os.path.exists(cache_dir)
        assert os.path.isdir(cache_dir)
    
    def test_persistence_preserves_lru_order(self, tmp_path):
        """Test that LRU order is preserved across save/load."""
        cache_dir = str(tmp_path / "cache")
        
        # Create cache with small size
        cache1 = CacheManager(embedding_cache_size=3, cache_dir=cache_dir)
        
        # Add 3 embeddings
        for i in range(3):
            cache1.set_embedding(f"text-{i}", "model", np.array([float(i)]))
        
        # Access text-0 to make it most recently used
        cache1.get_embedding("text-0", "model")
        
        # Save to disk
        cache1.save_to_disk()
        
        # Load in new instance
        cache2 = CacheManager(embedding_cache_size=3, cache_dir=cache_dir)
        cache2.load_from_disk()
        
        # Add a new item (should evict text-1, not text-0)
        cache2.set_embedding("text-3", "model", np.array([3.0]))
        
        # text-0 should still be cached (was accessed recently)
        assert cache2.get_embedding("text-0", "model") is not None
        
        # text-1 should be evicted (was LRU)
        assert cache2.get_embedding("text-1", "model") is None
    
    def test_file_io_error_includes_path(self, tmp_path):
        """Test that file I/O errors include the file path in the error message."""
        import os
        import sys
        
        # Skip on Windows as chmod doesn't work the same way
        if sys.platform == "win32":
            pytest.skip("chmod-based permission test not reliable on Windows")
        
        cache_dir = str(tmp_path / "cache")
        cache = CacheManager(cache_dir=cache_dir)
        cache.set_embedding("text", "model", np.array([0.1]))
        
        # Save successfully first
        cache.save_to_disk()
        
        # Make the cache directory read-only to trigger an error on save
        os.chmod(cache_dir, 0o444)
        
        try:
            # Try to save again (should fail)
            with pytest.raises(IOError) as exc_info:
                cache.save_to_disk()
            
            # Verify the error message includes the file path or cache dir
            error_message = str(exc_info.value)
            assert "embedding_cache.npz" in error_message or cache_dir in error_message
        finally:
            # Restore permissions for cleanup
            os.chmod(cache_dir, 0o755)

    def test_load_error_includes_path(self, tmp_path):
        """Test that load errors include the file path in the error message."""
        import os

        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)

        # Create a legacy pickle file — new code should detect and raise a clear error
        legacy_file = os.path.join(cache_dir, "embedding_cache.pkl")
        with open(legacy_file, "wb") as f:
            f.write(b"not valid pickle data")

        cache = CacheManager(cache_dir=cache_dir)

        # Try to load — should raise IOError mentioning the legacy file
        with pytest.raises(IOError) as exc_info:
            cache.load_from_disk()

        # Verify the error message includes the legacy file path
        error_message = str(exc_info.value)
        assert "embedding_cache.pkl" in error_message or cache_dir in error_message


# Property-Based Tests

class TestCacheProperties:
    """Property-based tests for CacheManager using Hypothesis."""

    @given(
        text=st.text(min_size=1, max_size=1000),
        model=st.text(min_size=1, max_size=100, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))),
        embedding_dim=st.integers(min_value=1, max_value=1024)
    )
    @settings(max_examples=100)
    def test_property_25_embedding_cache_hit_on_duplicate_content(self, text, model, embedding_dim):
        """
        Feature: chunker-improvements, Property 25: Embedding Cache Hit on Duplicate Content

        For any text that is embedded twice with the same model, the second embedding call
        should use the cached result (cache hit) and return the same embedding vector
        without recomputation.

        **Validates: Requirements 5.1**
        """
        cache = CacheManager(embedding_cache_size=10000)

        # Create a random embedding vector
        embedding = np.random.rand(embedding_dim).astype(np.float32)

        # First call: should be a cache miss
        result1 = cache.get_embedding(text, model)
        assert result1 is None, "First call should be a cache miss"

        # Get stats after first miss
        stats_after_miss = cache.get_stats()
        assert stats_after_miss["embedding_misses"] == 1, "Should have 1 miss after first call"
        assert stats_after_miss["embedding_hits"] == 0, "Should have 0 hits after first call"

        # Store the embedding
        cache.set_embedding(text, model, embedding)

        # Second call: should be a cache hit
        result2 = cache.get_embedding(text, model)
        assert result2 is not None, "Second call should be a cache hit"

        # Verify the cached embedding is identical
        assert np.array_equal(result2, embedding), "Cached embedding should match original"

        # Verify cache statistics show a hit
        stats_after_hit = cache.get_stats()
        assert stats_after_hit["embedding_hits"] == 1, "Should have 1 hit after second call"
        assert stats_after_hit["embedding_misses"] == 1, "Should still have 1 miss"

        # Third call: should also be a cache hit with same result
        result3 = cache.get_embedding(text, model)
        assert result3 is not None, "Third call should also be a cache hit"
        assert np.array_equal(result3, embedding), "Cached embedding should still match original"

        # Verify multiple hits increment correctly
        stats_after_second_hit = cache.get_stats()
        assert stats_after_second_hit["embedding_hits"] == 2, "Should have 2 hits after third call"
        assert stats_after_second_hit["embedding_misses"] == 1, "Should still have 1 miss"

        # Verify no recomputation occurred (same object reference or equal values)
        assert np.array_equal(result2, result3), "Multiple cache hits should return identical embeddings"


    @given(
        cache_size=st.integers(min_value=1, max_value=100),
        num_items=st.integers(min_value=2, max_value=200),
        model=st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))),
        embedding_dim=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=100)
    def test_property_26_lru_cache_eviction(self, cache_size, num_items, model, embedding_dim):
        """
        Feature: chunker-improvements, Property 26: LRU Cache Eviction
        
        For any cache with a maximum size of N, when N+1 items are added, the least
        recently used item should be evicted, and the cache size should remain at N.
        
        **Validates: Requirements 5.2**
        """
        # Skip if num_items <= cache_size (no eviction will occur)
        if num_items <= cache_size:
            return
        
        cache = CacheManager(embedding_cache_size=cache_size)
        
        # Store embeddings for later verification
        embeddings = {}
        
        # Add N items to fill the cache
        for i in range(cache_size):
            text = f"text-{i}"
            embedding = np.random.rand(embedding_dim).astype(np.float32)
            embeddings[text] = embedding
            cache.set_embedding(text, model, embedding)
        
        # Verify cache is at max size
        stats = cache.get_stats()
        assert stats["embedding_cache_size"] == cache_size, f"Cache should be at max size {cache_size}"
        
        # Add one more item (N+1), which should evict the LRU item (text-0)
        new_text = f"text-{cache_size}"
        new_embedding = np.random.rand(embedding_dim).astype(np.float32)
        embeddings[new_text] = new_embedding
        cache.set_embedding(new_text, model, new_embedding)
        
        # Verify cache size remains at N
        stats = cache.get_stats()
        assert stats["embedding_cache_size"] == cache_size, f"Cache size should remain at {cache_size} after eviction"
        
        # Verify the LRU item (text-0) was evicted
        result = cache.get_embedding("text-0", model)
        assert result is None, "LRU item (text-0) should have been evicted"
        
        # Verify the new item is cached
        result = cache.get_embedding(new_text, model)
        assert result is not None, "New item should be cached"
        assert np.array_equal(result, new_embedding), "New item should match the stored embedding"
        
        # Only test LRU access order update if cache_size >= 3
        # (need at least 3 items to test that accessing one prevents its eviction)
        if cache_size >= 3:
            # At this point, cache contains: [text-1, text-2, ..., text-{cache_size-1}, text-{cache_size}]
            # The order is: text-1 is LRU, text-{cache_size} is MRU
            
            # Access text-1 to make it most recently used
            # After this, order should be: text-2 is LRU, text-1 is MRU
            result = cache.get_embedding("text-1", model)
            assert result is not None, "text-1 should be cached before access test"
            
            # Add another item, which should evict text-2 (now the LRU), not text-1
            another_text = f"text-{cache_size + 1}"
            another_embedding = np.random.rand(embedding_dim).astype(np.float32)
            cache.set_embedding(another_text, model, another_embedding)
            
            # Verify cache size still at N
            stats = cache.get_stats()
            assert stats["embedding_cache_size"] == cache_size, f"Cache size should remain at {cache_size}"
            
            # Verify text-1 is still cached (was accessed recently)
            result = cache.get_embedding("text-1", model)
            assert result is not None, "text-1 should still be cached (was accessed recently)"
            
            # Verify text-2 was evicted (was LRU)
            result = cache.get_embedding("text-2", model)
            assert result is None, "text-2 should have been evicted (was LRU)"

    @given(
        text=st.text(min_size=1, max_size=1000),
        language=st.sampled_from([Language.ENGLISH, Language.ARABIC, Language.MIXED])
    )
    @settings(max_examples=100)
    def test_property_28_language_detection_cache_hit(self, text, language):
        """
        Feature: chunker-improvements, Property 28: Language Detection Cache Hit
        
        For any text that undergoes language detection twice, the second detection
        should use the cached result and return the same language without re-running
        detection.
        
        **Validates: Requirements 5.4**
        """
        cache = CacheManager(lang_detect_cache_size=5000)
        
        # First call: should be a cache miss
        result1 = cache.get_language(text)
        assert result1 is None, "First call should be a cache miss"
        
        # Get stats after first miss
        stats_after_miss = cache.get_stats()
        assert stats_after_miss["language_misses"] == 1, "Should have 1 miss after first call"
        assert stats_after_miss["language_hits"] == 0, "Should have 0 hits after first call"
        
        # Store the language detection result
        cache.set_language(text, language)
        
        # Second call: should be a cache hit
        result2 = cache.get_language(text)
        assert result2 is not None, "Second call should be a cache hit"
        
        # Verify the cached language is identical
        assert result2 == language, "Cached language should match original"
        
        # Verify cache statistics show a hit
        stats_after_hit = cache.get_stats()
        assert stats_after_hit["language_hits"] == 1, "Should have 1 hit after second call"
        assert stats_after_hit["language_misses"] == 1, "Should still have 1 miss"
        
        # Third call: should also be a cache hit with same result
        result3 = cache.get_language(text)
        assert result3 is not None, "Third call should also be a cache hit"
        assert result3 == language, "Cached language should still match original"
        
        # Verify multiple hits increment correctly
        stats_after_second_hit = cache.get_stats()
        assert stats_after_second_hit["language_hits"] == 2, "Should have 2 hits after third call"
        assert stats_after_second_hit["language_misses"] == 1, "Should still have 1 miss"
        
        # Verify no re-detection occurred (same result)
        assert result2 == result3, "Multiple cache hits should return identical language"

    @given(
        num_embeddings=st.integers(min_value=1, max_value=50),
        num_languages=st.integers(min_value=1, max_value=50),
        embedding_dim=st.integers(min_value=1, max_value=128),
        model=st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')))
    )
    @settings(max_examples=100)
    def test_property_27_cache_persistence_round_trip(self, num_embeddings, num_languages, embedding_dim, model):
        """
        Feature: chunker-improvements, Property 27: Cache Persistence Round-Trip
        
        For any cache with data, saving to disk and then loading from disk in a new
        instance should restore all cached entries with the same keys and values.
        
        **Validates: Requirements 5.3**
        """
        import tempfile
        import shutil
        
        # Create a temporary directory for this test
        temp_dir = tempfile.mkdtemp()
        
        try:
            cache_dir = temp_dir
            
            # Create first cache instance and populate it
            cache1 = CacheManager(
                embedding_cache_size=10000,
                lang_detect_cache_size=10000,
                cache_dir=cache_dir
            )
            
            # Store embeddings and their expected values
            embedding_data = {}
            for i in range(num_embeddings):
                text = f"embedding-text-{i}"
                embedding = np.random.rand(embedding_dim).astype(np.float32)
                embedding_data[text] = embedding
                cache1.set_embedding(text, model, embedding)
            
            # Store language detections and their expected values
            language_data = {}
            languages = [Language.ENGLISH, Language.ARABIC, Language.MIXED]
            for i in range(num_languages):
                text = f"language-text-{i}"
                language = languages[i % len(languages)]
                language_data[text] = language
                cache1.set_language(text, language)
            
            # Access some entries to create hit statistics
            if num_embeddings > 0:
                cache1.get_embedding("embedding-text-0", model)
            if num_languages > 0:
                cache1.get_language("language-text-0")
            
            # Get statistics before save
            stats_before = cache1.get_stats()
            
            # Save to disk
            cache1.save_to_disk()
            
            # Create a new cache instance and load from disk
            cache2 = CacheManager(
                embedding_cache_size=10000,
                lang_detect_cache_size=10000,
                cache_dir=cache_dir
            )
            cache2.load_from_disk()
            
            # Verify statistics are restored immediately after load (before any access)
            stats_after_load = cache2.get_stats()
            assert stats_after_load["embedding_hits"] == stats_before["embedding_hits"], \
                "Embedding hit count should be restored"
            assert stats_after_load["embedding_misses"] == stats_before["embedding_misses"], \
                "Embedding miss count should be restored"
            assert stats_after_load["language_hits"] == stats_before["language_hits"], \
                "Language hit count should be restored"
            assert stats_after_load["language_misses"] == stats_before["language_misses"], \
                "Language miss count should be restored"
            
            # Verify cache sizes match
            assert stats_after_load["embedding_cache_size"] == len(embedding_data), \
                "Embedding cache size should match number of stored embeddings"
            assert stats_after_load["language_cache_size"] == len(language_data), \
                "Language cache size should match number of stored languages"
            
            # Verify all embeddings are restored with correct values
            for text, expected_embedding in embedding_data.items():
                result = cache2.get_embedding(text, model)
                assert result is not None, f"Embedding for '{text}' should be restored"
                assert np.array_equal(result, expected_embedding), \
                    f"Restored embedding for '{text}' should match original"
            
            # Verify all language detections are restored with correct values
            for text, expected_language in language_data.items():
                result = cache2.get_language(text)
                assert result is not None, f"Language for '{text}' should be restored"
                assert result == expected_language, \
                    f"Restored language for '{text}' should match original"
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

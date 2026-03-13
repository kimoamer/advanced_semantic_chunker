"""
Unit tests for new ChunkerConfig validation fields.

Tests the validation logic for caching, batch processing, error handling,
and observability configuration fields added in task 12.1.
"""

import unittest

from chunker.config import ChunkerConfig, StrategyType


class TestCachingConfigValidation(unittest.TestCase):
    """Test validation of caching configuration fields."""

    def test_negative_embedding_cache_size_raises_error(self):
        """Test that negative embedding_cache_size raises ValueError."""
        config = ChunkerConfig(embedding_cache_size=-1)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("embedding_cache_size must be >= 0", str(ctx.exception))

    def test_negative_language_cache_size_raises_error(self):
        """Test that negative language_cache_size raises ValueError."""
        config = ChunkerConfig(language_cache_size=-1)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("language_cache_size must be >= 0", str(ctx.exception))

    def test_persist_cache_without_cache_dir_raises_error(self):
        """Test that persist_cache_to_disk=True without cache_dir raises ValueError."""
        config = ChunkerConfig(persist_cache_to_disk=True, cache_dir=None)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("cache_dir must be specified", str(ctx.exception))

    def test_persist_cache_with_cache_dir_is_valid(self):
        """Test that persist_cache_to_disk=True with cache_dir is valid."""
        config = ChunkerConfig(persist_cache_to_disk=True, cache_dir="/tmp/cache")
        config.validate()  # Should not raise

    def test_zero_cache_sizes_are_valid(self):
        """Test that zero cache sizes are valid (disables caching)."""
        config = ChunkerConfig(
            embedding_cache_size=0,
            language_cache_size=0
        )
        config.validate()  # Should not raise


class TestBatchProcessingConfigValidation(unittest.TestCase):
    """Test validation of batch processing configuration fields."""

    def test_zero_batch_size_raises_error(self):
        """Test that batch_size=0 raises ValueError."""
        config = ChunkerConfig(batch_size=0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("batch_size must be >= 1", str(ctx.exception))

    def test_negative_batch_size_raises_error(self):
        """Test that negative batch_size raises ValueError."""
        config = ChunkerConfig(batch_size=-5)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("batch_size must be >= 1", str(ctx.exception))

    def test_zero_num_workers_raises_error(self):
        """Test that num_workers=0 raises ValueError."""
        config = ChunkerConfig(num_workers=0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("num_workers must be >= 1 or None", str(ctx.exception))

    def test_negative_num_workers_raises_error(self):
        """Test that negative num_workers raises ValueError."""
        config = ChunkerConfig(num_workers=-2)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("num_workers must be >= 1 or None", str(ctx.exception))

    def test_none_num_workers_is_valid(self):
        """Test that num_workers=None is valid (uses CPU count)."""
        config = ChunkerConfig(num_workers=None)
        config.validate()  # Should not raise

    def test_positive_num_workers_is_valid(self):
        """Test that positive num_workers is valid."""
        config = ChunkerConfig(num_workers=4)
        config.validate()  # Should not raise


class TestErrorHandlingConfigValidation(unittest.TestCase):
    """Test validation of error handling configuration fields."""

    def test_negative_max_retries_raises_error(self):
        """Test that negative max_retries raises ValueError."""
        config = ChunkerConfig(max_retries=-1)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("max_retries must be >= 0", str(ctx.exception))

    def test_zero_max_retries_is_valid(self):
        """Test that max_retries=0 is valid (no retries)."""
        config = ChunkerConfig(max_retries=0)
        config.validate()  # Should not raise

    def test_zero_retry_backoff_factor_raises_error(self):
        """Test that retry_backoff_factor=0 raises ValueError."""
        config = ChunkerConfig(retry_backoff_factor=0.0)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("retry_backoff_factor must be > 0", str(ctx.exception))

    def test_negative_retry_backoff_factor_raises_error(self):
        """Test that negative retry_backoff_factor raises ValueError."""
        config = ChunkerConfig(retry_backoff_factor=-1.5)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("retry_backoff_factor must be > 0", str(ctx.exception))

    def test_positive_retry_backoff_factor_is_valid(self):
        """Test that positive retry_backoff_factor is valid."""
        config = ChunkerConfig(retry_backoff_factor=2.0)
        config.validate()  # Should not raise


class TestObservabilityConfigValidation(unittest.TestCase):
    """Test validation of observability configuration fields."""

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log_level raises ValueError."""
        config = ChunkerConfig(log_level="INVALID")
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("log_level must be one of", str(ctx.exception))

    def test_valid_log_levels(self):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = ChunkerConfig(log_level=level)
            config.validate()  # Should not raise

    def test_lowercase_log_level_raises_error(self):
        """Test that lowercase log level raises ValueError."""
        config = ChunkerConfig(log_level="info")
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("log_level must be one of", str(ctx.exception))


class TestDefaultConfigValues(unittest.TestCase):
    """Test that default values for new fields are correct."""

    def test_caching_defaults(self):
        """Test default values for caching configuration."""
        config = ChunkerConfig()
        self.assertTrue(config.enable_embedding_cache)
        self.assertEqual(config.embedding_cache_size, 10000)
        self.assertTrue(config.enable_language_cache)
        self.assertEqual(config.language_cache_size, 5000)
        self.assertFalse(config.persist_cache_to_disk)
        self.assertIsNone(config.cache_dir)

    def test_batch_processing_defaults(self):
        """Test default values for batch processing configuration."""
        config = ChunkerConfig()
        self.assertTrue(config.batch_embedding_calls)
        self.assertEqual(config.batch_size, 32)
        self.assertFalse(config.enable_parallel_processing)
        self.assertIsNone(config.num_workers)

    def test_error_handling_defaults(self):
        """Test default values for error handling configuration."""
        config = ChunkerConfig()
        self.assertTrue(config.enable_fallbacks)
        self.assertTrue(config.retry_on_network_error)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_backoff_factor, 2.0)

    def test_observability_defaults(self):
        """Test default values for observability configuration."""
        config = ChunkerConfig()
        self.assertTrue(config.enable_metrics)
        self.assertTrue(config.enable_structured_logging)
        self.assertEqual(config.log_level, "INFO")
        self.assertIsNone(config.progress_callback)

    def test_lazy_loading_defaults(self):
        """Test default values for lazy loading configuration."""
        config = ChunkerConfig()
        self.assertEqual(config.preload_components, [])
        self.assertTrue(config.lazy_load_embeddings)
        self.assertTrue(config.lazy_load_nlp_tools)


class TestConfigWithAllNewFields(unittest.TestCase):
    """Test configuration with all new fields set to non-default values."""

    def test_valid_config_with_all_new_fields(self):
        """Test that a config with all new fields set is valid."""
        config = ChunkerConfig(
            # Caching
            enable_embedding_cache=False,
            embedding_cache_size=5000,
            enable_language_cache=False,
            language_cache_size=2500,
            persist_cache_to_disk=True,
            cache_dir="/tmp/chunker_cache",
            # Batch processing
            batch_embedding_calls=False,
            batch_size=64,
            enable_parallel_processing=True,
            num_workers=8,
            # Error handling
            enable_fallbacks=False,
            retry_on_network_error=False,
            max_retries=5,
            retry_backoff_factor=1.5,
            # Observability
            enable_metrics=False,
            enable_structured_logging=False,
            log_level="DEBUG",
            progress_callback=lambda x: None,
            # Lazy loading
            preload_components=["embeddings", "language_processors"],
            lazy_load_embeddings=False,
            lazy_load_nlp_tools=False,
        )
        config.validate()  # Should not raise


if __name__ == "__main__":
    unittest.main()

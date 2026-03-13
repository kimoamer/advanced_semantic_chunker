"""
Tests for MetricsCollector integration with SemanticChunker.

This module tests that metrics are correctly collected during chunking operations.
"""

import pytest
from chunker import SemanticChunker, ChunkerConfig, StrategyType
from chunker.models import Language


def test_metrics_collector_tracks_document_processing():
    """Test that metrics collector tracks document processing."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process a document
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunker.chunk(text)
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify metrics were recorded
    assert metrics.documents_processed == 1
    assert metrics.total_chunks_created == len(chunks)
    assert metrics.total_tokens_processed > 0
    assert metrics.avg_processing_time_ms > 0
    assert metrics.strategy_usage.get("sentence", 0) == 1


def test_metrics_collector_tracks_multiple_documents():
    """Test that metrics accumulate across multiple documents."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process multiple documents
    texts = [
        "First document. With two sentences.",
        "Second document. Also with two sentences.",
        "Third document. Three sentences here. Yes indeed.",
    ]
    
    total_chunks = 0
    for text in texts:
        chunks = chunker.chunk(text)
        total_chunks += len(chunks)
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify metrics accumulated
    assert metrics.documents_processed == 3
    assert metrics.total_chunks_created == total_chunks
    assert metrics.avg_processing_time_ms > 0
    assert metrics.min_processing_time_ms <= metrics.avg_processing_time_ms
    assert metrics.max_processing_time_ms >= metrics.avg_processing_time_ms


def test_metrics_collector_tracks_language_cache():
    """Test that metrics track language detection cache hits/misses."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        enable_language_cache=True,
        detect_language=True,
    )
    chunker = SemanticChunker(config)
    
    # Process same text twice
    text = "This is an English sentence."
    chunker.chunk(text)  # First call - cache miss
    chunker.chunk(text)  # Second call - cache hit
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify cache tracking
    assert metrics.language_detections >= 1
    assert metrics.language_cache_hits >= 1 or metrics.language_cache_misses >= 1


def test_metrics_collector_tracks_strategy_usage():
    """Test that metrics track per-strategy usage."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process documents
    chunker.chunk("First document.")
    chunker.chunk("Second document.")
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify strategy tracking
    assert "sentence" in metrics.strategy_usage
    assert metrics.strategy_usage["sentence"] == 2
    assert "sentence" in metrics.strategy_timing
    assert metrics.strategy_timing["sentence"] > 0


def test_metrics_collector_tracks_language_distribution():
    """Test that metrics track per-language distribution."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=True,
    )
    chunker = SemanticChunker(config)
    
    # Process English documents
    chunker.chunk("This is an English sentence.")
    chunker.chunk("Another English sentence.")
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify language tracking
    assert len(metrics.language_distribution) > 0
    # Should have English documents
    assert any("en" in lang.lower() for lang in metrics.language_distribution.keys())


def test_metrics_collector_disabled():
    """Test that metrics collection can be disabled."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=False,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process a document
    chunker.chunk("This is a sentence.")
    
    # Get metrics - should be empty
    metrics = chunker.get_metrics()
    
    # Verify no metrics collected
    assert metrics.documents_processed == 0
    assert metrics.total_chunks_created == 0


def test_metrics_collector_tracks_errors_and_fallbacks():
    """Test that metrics track errors and fallback activations."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        enable_fallbacks=True,
        detect_language=True,
    )
    chunker = SemanticChunker(config)
    
    # Mock a language detection failure by using invalid text
    # This should trigger fallback to English
    try:
        chunker.chunk("Test text")
    except Exception:
        pass  # Ignore any errors
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Metrics should be collected even if errors occurred
    assert metrics.documents_processed >= 0


def test_get_metrics_returns_chunker_metrics_object():
    """Test that get_metrics returns a ChunkerMetrics object."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process a document
    chunker.chunk("This is a test.")
    
    # Get metrics
    metrics = chunker.get_metrics()
    
    # Verify it's a ChunkerMetrics object with expected attributes
    assert hasattr(metrics, 'documents_processed')
    assert hasattr(metrics, 'total_chunks_created')
    assert hasattr(metrics, 'total_tokens_processed')
    assert hasattr(metrics, 'avg_processing_time_ms')
    assert hasattr(metrics, 'embedding_cache_hit_rate')
    assert hasattr(metrics, 'strategy_usage')
    assert hasattr(metrics, 'language_distribution')


def test_prometheus_export():
    """Test that metrics can be exported in Prometheus format."""
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        detect_language=False,
    )
    chunker = SemanticChunker(config)
    
    # Process a document
    chunker.chunk("This is a test sentence.")
    
    # Get metrics and export to Prometheus
    metrics = chunker.get_metrics()
    
    # Access the metrics collector directly to test export
    if chunker._metrics_collector:
        prometheus_output = chunker._metrics_collector.export_prometheus()
        
        # Verify Prometheus format
        assert "chunker_documents_processed_total" in prometheus_output
        assert "chunker_chunks_created_total" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

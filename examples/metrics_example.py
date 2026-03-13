"""
Example demonstrating metrics collection in SemanticChunker.

This example shows how to:
1. Enable metrics collection
2. Process documents
3. Access metrics summary
4. Export metrics in Prometheus format
"""

from chunker import SemanticChunker, ChunkerConfig, StrategyType


def main():
    # Configure chunker with metrics enabled
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_metrics=True,
        enable_embedding_cache=True,
        enable_language_cache=True,
        detect_language=True,
        verbose=True,
    )
    
    chunker = SemanticChunker(config)
    
    # Process multiple documents
    documents = [
        "This is the first document. It has multiple sentences. Each sentence will be chunked.",
        "Second document here. Also with multiple sentences. Testing metrics collection.",
        "Third document. More sentences to process. Metrics should accumulate.",
        "This is the first document. It has multiple sentences. Each sentence will be chunked.",  # Duplicate for cache testing
    ]
    
    print("Processing documents...\n")
    for i, doc in enumerate(documents, 1):
        chunks = chunker.chunk(doc, document_id=f"doc_{i}")
        print(f"Document {i}: {len(chunks)} chunks created")
    
    # Get metrics summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    metrics = chunker.get_metrics()
    
    print(f"\nDocument Processing:")
    print(f"  Documents processed: {metrics.documents_processed}")
    print(f"  Total chunks created: {metrics.total_chunks_created}")
    print(f"  Total tokens processed: {metrics.total_tokens_processed}")
    
    print(f"\nTiming Statistics:")
    print(f"  Average processing time: {metrics.avg_processing_time_ms:.2f} ms")
    print(f"  Min processing time: {metrics.min_processing_time_ms:.2f} ms")
    print(f"  Max processing time: {metrics.max_processing_time_ms:.2f} ms")
    print(f"  Total processing time: {metrics.total_processing_time_ms:.2f} ms")
    
    print(f"\nCache Performance:")
    print(f"  Embedding calls: {metrics.embedding_calls}")
    print(f"  Embedding cache hits: {metrics.embedding_cache_hits}")
    print(f"  Embedding cache misses: {metrics.embedding_cache_misses}")
    print(f"  Embedding cache hit rate: {metrics.embedding_cache_hit_rate:.2%}")
    print(f"  Language detections: {metrics.language_detections}")
    print(f"  Language cache hits: {metrics.language_cache_hits}")
    print(f"  Language cache misses: {metrics.language_cache_misses}")
    
    print(f"\nError Tracking:")
    print(f"  Total errors: {metrics.total_errors}")
    print(f"  Recoverable errors: {metrics.recoverable_errors}")
    print(f"  Unrecoverable errors: {metrics.unrecoverable_errors}")
    print(f"  Fallbacks triggered: {metrics.fallbacks_triggered}")
    
    print(f"\nStrategy Usage:")
    for strategy, count in metrics.strategy_usage.items():
        avg_time = metrics.strategy_timing.get(strategy, 0)
        print(f"  {strategy}: {count} documents, avg {avg_time:.2f} ms")
    
    print(f"\nLanguage Distribution:")
    for language, count in metrics.language_distribution.items():
        print(f"  {language}: {count} documents")
    
    # Export to Prometheus format
    print("\n" + "="*60)
    print("PROMETHEUS EXPORT")
    print("="*60)
    
    prometheus_output = chunker._metrics_collector.export_prometheus()
    print(prometheus_output)
    
    # Get cache statistics
    print("="*60)
    print("CACHE STATISTICS")
    print("="*60)
    
    cache_stats = chunker.get_cache_stats()
    if cache_stats:
        print(f"\nEmbedding Cache:")
        print(f"  Size: {cache_stats.get('embedding_cache_size', 0)}")
        print(f"  Hits: {cache_stats.get('embedding_cache_hits', 0)}")
        print(f"  Misses: {cache_stats.get('embedding_cache_misses', 0)}")
        print(f"  Hit rate: {cache_stats.get('embedding_cache_hit_rate', 0):.2%}")
        
        print(f"\nLanguage Cache:")
        print(f"  Size: {cache_stats.get('language_cache_size', 0)}")
        print(f"  Hits: {cache_stats.get('language_cache_hits', 0)}")
        print(f"  Misses: {cache_stats.get('language_cache_misses', 0)}")
        print(f"  Hit rate: {cache_stats.get('language_cache_hit_rate', 0):.2%}")


if __name__ == "__main__":
    main()

"""
Example demonstrating BatchProcessor usage.

This example shows how to use the BatchProcessor class for efficient
batch processing of multiple documents with progress tracking and
parallel processing support.
"""

from chunker import SemanticChunker, ChunkerConfig, StrategyType, BatchProcessor


def basic_batch_processing():
    """Basic batch processing example."""
    print("=" * 60)
    print("Basic Batch Processing")
    print("=" * 60)
    
    # Create chunker with simple sentence strategy
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        chunk_size=100,
        chunk_overlap=20,
    )
    chunker = SemanticChunker(config)
    
    # Create batch processor
    processor = BatchProcessor(chunker)
    
    # Process multiple documents
    texts = [
        "This is the first document. It contains multiple sentences. Each sentence provides valuable information.",
        "This is the second document. It has different content. The chunker will process it efficiently.",
        "This is the third document. More content here. Testing batch processing capabilities.",
    ]
    
    results = processor.process_batch(texts)
    
    print(f"\nProcessed {len(results)} documents")
    for i, chunks in enumerate(results):
        print(f"  Document {i+1}: {len(chunks)} chunks")


def batch_processing_with_progress():
    """Batch processing with progress callback."""
    print("\n" + "=" * 60)
    print("Batch Processing with Progress Tracking")
    print("=" * 60)
    
    config = ChunkerConfig(strategy=StrategyType.SENTENCE)
    chunker = SemanticChunker(config)
    
    # Define progress callback
    def progress_callback(current, total):
        percentage = (current / total) * 100
        print(f"  Progress: {current}/{total} ({percentage:.1f}%)")
    
    # Create processor with progress callback
    processor = BatchProcessor(chunker, progress_callback=progress_callback)

    
    # Process documents with progress tracking
    texts = [f"Document {i} with some content for processing." for i in range(1, 6)]
    
    print("\nProcessing documents:")
    results = processor.process_batch(texts)
    print(f"\nCompleted! Processed {len(results)} documents")


def parallel_batch_processing():
    """Parallel batch processing example."""
    print("\n" + "=" * 60)
    print("Parallel Batch Processing")
    print("=" * 60)
    
    config = ChunkerConfig(strategy=StrategyType.SENTENCE)
    chunker = SemanticChunker(config)
    processor = BatchProcessor(chunker)
    
    # Create a larger batch of documents
    texts = [
        f"Document {i}: This is a sample document with multiple sentences. "
        f"It contains various information. The content is diverse and interesting."
        for i in range(1, 11)
    ]
    
    print(f"\nProcessing {len(texts)} documents in parallel...")
    
    # Process with parallel execution
    results = processor.process_batch(texts, parallel=True, num_workers=4)
    
    print(f"Completed! Processed {len(results)} documents")
    total_chunks = sum(len(chunks) for chunks in results)
    print(f"Total chunks created: {total_chunks}")


def streaming_batch_processing():
    """Streaming batch processing with iterator."""
    print("\n" + "=" * 60)
    print("Streaming Batch Processing")
    print("=" * 60)
    
    config = ChunkerConfig(strategy=StrategyType.SENTENCE)
    chunker = SemanticChunker(config)
    processor = BatchProcessor(chunker)
    
    texts = [
        "First document content with multiple sentences.",
        "Second document with different information.",
        "Third document with more data to process.",
    ]
    
    print("\nProcessing documents one at a time:")
    for index, chunks in processor.process_with_progress(texts):
        print(f"  Document {index+1}: {len(chunks)} chunks created")
        # Process chunks immediately (streaming)
        for chunk in chunks:
            print(f"    - Chunk: {chunk.text[:50]}...")


def batch_processing_with_source_files():
    """Batch processing with source file tracking."""
    print("\n" + "=" * 60)
    print("Batch Processing with Source Files")
    print("=" * 60)
    
    config = ChunkerConfig(strategy=StrategyType.SENTENCE)
    chunker = SemanticChunker(config)
    processor = BatchProcessor(chunker)
    
    texts = [
        "Content from the first file.",
        "Content from the second file.",
        "Content from the third file.",
    ]
    source_files = ["document1.txt", "document2.txt", "document3.txt"]
    
    results = processor.process_batch(texts, source_files=source_files)
    
    print("\nProcessed documents with source tracking:")
    for chunks in results:
        if chunks:
            source = chunks[0].metadata.source_file
            print(f"  {source}: {len(chunks)} chunks")


if __name__ == "__main__":
    # Run all examples
    basic_batch_processing()
    batch_processing_with_progress()
    parallel_batch_processing()
    streaming_batch_processing()
    batch_processing_with_source_files()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

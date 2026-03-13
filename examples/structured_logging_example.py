"""
Example demonstrating structured logging in SemanticChunker.

This example shows how to:
1. Enable structured logging
2. Configure log levels
3. View structured log output with timing information
4. Monitor key events during chunking
"""

import logging
from chunker import SemanticChunker, ChunkerConfig, StrategyType

# Configure logging to see structured output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Just show the message (which will be JSON)
)

def main():
    print("=" * 80)
    print("Structured Logging Example")
    print("=" * 80)
    
    # Example 1: Basic structured logging
    print("\n1. Basic Structured Logging (INFO level)")
    print("-" * 80)
    
    config = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_structured_logging=True,
        log_level="INFO",
        detect_language=True
    )
    
    chunker = SemanticChunker(config)
    
    text = """
    Machine learning is transforming the world. It enables computers to learn from data.
    Deep learning is a subset of machine learning. Neural networks are the foundation.
    """
    
    chunks = chunker.chunk(text, document_id="ml_doc")
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Example 2: Debug level logging (includes cache hits)
    print("\n\n2. Debug Level Logging (includes cache operations)")
    print("-" * 80)
    
    config_debug = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_structured_logging=True,
        log_level="DEBUG",
        enable_embedding_cache=True,
        detect_language=True
    )
    
    chunker_debug = SemanticChunker(config_debug)
    
    # Chunk the same text twice to see cache hits
    chunks1 = chunker_debug.chunk(text, document_id="cache_test_1")
    chunks2 = chunker_debug.chunk(text, document_id="cache_test_2")
    
    print(f"\nFirst run: {len(chunks1)} chunks")
    print(f"Second run: {len(chunks2)} chunks (should use cached language detection)")
    
    # Example 3: Disabled structured logging
    print("\n\n3. Structured Logging Disabled")
    print("-" * 80)
    
    config_disabled = ChunkerConfig(
        strategy=StrategyType.SENTENCE,
        enable_structured_logging=False,
        verbose=True  # Use verbose mode instead
    )
    
    chunker_disabled = SemanticChunker(config_disabled)
    chunks3 = chunker_disabled.chunk(text, document_id="no_structured_log")
    
    print(f"\nCreated {len(chunks3)} chunks (with verbose logging)")
    
    # Example 4: Timing information
    print("\n\n4. Timing Information in Logs")
    print("-" * 80)
    print("Notice the 'duration_ms' fields in the JSON logs above.")
    print("These show timing for:")
    print("  - Overall chunking operation (chunking_complete event)")
    print("  - Strategy execution (operation_timing event)")
    
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("\nKey observations:")
    print("1. Structured logs are in JSON format for easy parsing")
    print("2. Each event has a timestamp, component, and event type")
    print("3. Timing information is included for performance monitoring")
    print("4. Language detection results are logged")
    print("5. Log level can be configured (DEBUG, INFO, WARNING, ERROR)")


if __name__ == "__main__":
    main()

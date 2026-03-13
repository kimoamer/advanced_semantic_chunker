"""
Property-based tests for embedding memory cleanup using Hypothesis.

These tests verify that intermediate embedding storage is cleared or empty
after chunking operations complete, ensuring no memory leaks from retained embeddings.
"""

import gc
import sys
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from chunker.config import ChunkerConfig, EmbeddingProvider, StrategyType
from chunker.core import SemanticChunker


# Custom strategies for generating test data
@st.composite
def valid_text_for_semantic_chunking(draw):
    """Generate valid text for semantic chunking (multiple sentences)."""
    # Generate text with multiple distinct sentences to trigger embedding generation
    # Use words instead of random characters to create more realistic sentences
    word_list = ["hello", "world", "test", "data", "example", "sentence", "chunk", 
                 "semantic", "analysis", "processing", "document", "text", "content",
                 "information", "system", "application", "feature", "function", "method"]
    
    sentences = []
    num_sentences = draw(st.integers(min_value=5, max_value=15))
    
    for _ in range(num_sentences):
        # Generate a sentence with 5-15 words
        num_words = draw(st.integers(min_value=5, max_value=15))
        words = [draw(st.sampled_from(word_list)) for _ in range(num_words)]
        sentence = " ".join(words).capitalize()
        sentences.append(sentence)
    
    return ". ".join(sentences) + "."


class MockEmbeddingProvider:
    """Mock embedding provider that tracks embedding storage."""
    
    def __init__(self):
        self.embeddings_generated = []
        self.embed_call_count = 0
    
    def embed(self, texts):
        """Generate mock embeddings and track them."""
        if isinstance(texts, str):
            texts = [texts]
        
        self.embed_call_count += 1
        
        # Generate mock embeddings (384-dimensional vectors)
        embeddings = np.random.rand(len(texts), 384).astype(np.float32)
        
        # Track the embeddings we generated
        self.embeddings_generated.append(embeddings)
        
        return embeddings
    
    def get_total_embedding_memory(self):
        """Calculate total memory used by stored embeddings."""
        total_bytes = 0
        for emb in self.embeddings_generated:
            total_bytes += emb.nbytes
        return total_bytes
    
    def clear_embeddings(self):
        """Clear all stored embeddings."""
        self.embeddings_generated.clear()


class TestEmbeddingMemoryCleanupProperties:
    """Property-based tests for embedding memory cleanup."""

    @given(
        text=valid_text_for_semantic_chunking()
    )
    @settings(max_examples=100)
    def test_property_5_embedding_memory_cleanup(self, text):
        """
        Feature: chunker-improvements, Property 5: Embedding Memory Cleanup
        
        For any completed chunking operation, intermediate embedding storage should
        be cleared or empty after the operation completes.
        
        **Validates: Requirements 1.8**
        """
        # Create config with semantic strategy (requires embeddings)
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            chunk_size=512,
            chunk_overlap=0,
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_embedding_cache=False,  # Disable cache to ensure embeddings are generated
            enable_metrics=False,
            enable_structured_logging=False,
        )
        
        # Create mock embedding provider to track memory
        mock_provider = MockEmbeddingProvider()
        
        # Create chunker with mock provider
        chunker = SemanticChunker(config, embedding_provider=mock_provider)
        
        # Start memory tracking
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Perform chunking operation
        chunks = chunker.chunk(text)
        
        # Verify chunks were created
        assert len(chunks) > 0, "Chunking should produce at least one chunk"
        
        # Verify embeddings were generated during chunking
        assert mock_provider.embed_call_count > 0, "Embeddings should have been generated"
        
        # Take snapshot after chunking
        snapshot_after = tracemalloc.take_snapshot()
        
        # Force garbage collection to clean up any unreferenced objects
        gc.collect()
        
        # Take snapshot after garbage collection
        snapshot_after_gc = tracemalloc.take_snapshot()
        
        # Stop memory tracking
        tracemalloc.stop()
        
        # Check that the SemanticStrategy doesn't retain embeddings
        # The strategy should not have any instance variables storing embeddings
        strategy = chunker._strategy
        if strategy is not None:
            # Check that strategy doesn't have embedding storage attributes
            strategy_dict = vars(strategy)
            
            # Look for any numpy arrays that might be embeddings
            for attr_name, attr_value in strategy_dict.items():
                if isinstance(attr_value, np.ndarray):
                    # If there's a numpy array, it should be small (not embedding storage)
                    # Embeddings would be large (N x 384 or similar)
                    if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                        pytest.fail(
                            f"Strategy retains large numpy array '{attr_name}' "
                            f"with shape {attr_value.shape} after chunking. "
                            f"This suggests embeddings are not being cleaned up."
                        )
        
        # Verify that the chunker itself doesn't retain embeddings
        chunker_dict = vars(chunker)
        for attr_name, attr_value in chunker_dict.items():
            if isinstance(attr_value, np.ndarray):
                if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                    pytest.fail(
                        f"Chunker retains large numpy array '{attr_name}' "
                        f"with shape {attr_value.shape} after chunking. "
                        f"This suggests embeddings are not being cleaned up."
                    )
        
        # Additional check: verify no large numpy arrays in memory
        # by checking that the mock provider's embeddings are the only ones
        # (and they should be cleared or not retained by the chunker)
        
        # The key assertion: after chunking completes, the chunker and strategy
        # should not hold references to the intermediate embeddings
        # This is verified by the checks above
        
        # Success: no embedding storage found in chunker or strategy after completion

    @given(
        texts=st.lists(
            valid_text_for_semantic_chunking(),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=50)
    def test_property_5_embedding_memory_cleanup_batch(self, texts):
        """
        Feature: chunker-improvements, Property 5: Embedding Memory Cleanup (Batch)
        
        For any batch of completed chunking operations, intermediate embedding storage
        should be cleared or empty after each operation completes.
        
        **Validates: Requirements 1.8**
        """
        # Create config with semantic strategy
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            chunk_size=512,
            chunk_overlap=0,
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_embedding_cache=False,  # Disable cache
            enable_metrics=False,
            enable_structured_logging=False,
        )
        
        # Create mock embedding provider
        mock_provider = MockEmbeddingProvider()
        
        # Create chunker
        chunker = SemanticChunker(config, embedding_provider=mock_provider)
        
        # Process each text
        for text in texts:
            # Clear mock provider's tracking before each operation
            initial_call_count = mock_provider.embed_call_count
            
            # Perform chunking
            chunks = chunker.chunk(text)
            
            # Verify chunks were created
            assert len(chunks) > 0, "Chunking should produce at least one chunk"
            
            # Verify embeddings were generated
            assert mock_provider.embed_call_count > initial_call_count, \
                "Embeddings should have been generated for this text"
            
            # Force garbage collection
            gc.collect()
            
            # Check that strategy doesn't retain embeddings after this operation
            strategy = chunker._strategy
            if strategy is not None:
                strategy_dict = vars(strategy)
                for attr_name, attr_value in strategy_dict.items():
                    if isinstance(attr_value, np.ndarray):
                        if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                            pytest.fail(
                                f"Strategy retains embeddings after operation {texts.index(text) + 1}. "
                                f"Attribute '{attr_name}' has shape {attr_value.shape}."
                            )
        
        # After all operations, verify no embedding storage remains
        chunker_dict = vars(chunker)
        for attr_name, attr_value in chunker_dict.items():
            if isinstance(attr_value, np.ndarray):
                if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                    pytest.fail(
                        f"Chunker retains embeddings after all operations. "
                        f"Attribute '{attr_name}' has shape {attr_value.shape}."
                    )

    @given(
        text=valid_text_for_semantic_chunking()
    )
    @settings(max_examples=50)
    def test_property_5_embedding_memory_cleanup_with_cache(self, text):
        """
        Feature: chunker-improvements, Property 5: Embedding Memory Cleanup (With Cache)
        
        Even with caching enabled, intermediate embedding storage in the strategy
        should be cleared after chunking completes. Only the cache should retain embeddings.
        
        **Validates: Requirements 1.8**
        """
        # Create config with semantic strategy and cache enabled
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            chunk_size=512,
            chunk_overlap=0,
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMER,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_embedding_cache=True,  # Enable cache
            embedding_cache_size=1000,
            enable_metrics=False,
            enable_structured_logging=False,
        )
        
        # Create mock embedding provider
        mock_provider = MockEmbeddingProvider()
        
        # Create chunker
        chunker = SemanticChunker(config, embedding_provider=mock_provider)
        
        # Perform chunking
        chunks = chunker.chunk(text)
        
        # Verify chunks were created
        assert len(chunks) > 0, "Chunking should produce at least one chunk"
        
        # Force garbage collection
        gc.collect()
        
        # Check that strategy doesn't retain embeddings
        # (cache is separate from strategy's intermediate storage)
        strategy = chunker._strategy
        if strategy is not None:
            strategy_dict = vars(strategy)
            
            # Exclude the cache_manager from this check (it's supposed to store embeddings)
            for attr_name, attr_value in strategy_dict.items():
                if attr_name == 'cache_manager':
                    continue  # Cache is allowed to store embeddings
                
                if isinstance(attr_value, np.ndarray):
                    if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                        pytest.fail(
                            f"Strategy retains embeddings in '{attr_name}' "
                            f"with shape {attr_value.shape}. "
                            f"Only cache should store embeddings, not strategy itself."
                        )
        
        # Verify cache is working (it should have stored embeddings)
        if chunker._cache_manager:
            cache_stats = chunker.get_cache_stats()
            # Cache should have some entries after chunking
            assert cache_stats.get('embedding_cache_size', 0) > 0, \
                "Cache should contain embeddings after chunking"


class TestEmbeddingMemoryCleanupEdgeCases:
    """Unit tests for edge cases in embedding memory cleanup."""
    
    def test_empty_text_no_embedding_storage(self):
        """Empty text should not create any embedding storage."""
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            enable_embedding_cache=False,
            enable_metrics=False,
            enable_structured_logging=False,
        )
        
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticChunker(config, embedding_provider=mock_provider)
        
        # Chunk empty text
        chunks = chunker.chunk("")
        
        # Should return empty list
        assert len(chunks) == 0
        
        # No embeddings should have been generated
        assert mock_provider.embed_call_count == 0
        
        # No embedding storage in strategy
        if chunker._strategy:
            strategy_dict = vars(chunker._strategy)
            for attr_value in strategy_dict.values():
                if isinstance(attr_value, np.ndarray):
                    if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                        pytest.fail("Strategy should not have embedding storage for empty text")
    
    def test_single_sentence_no_embedding_storage(self):
        """Single sentence should not retain embedding storage after chunking."""
        config = ChunkerConfig(
            strategy=StrategyType.SEMANTIC,
            enable_embedding_cache=False,
            enable_metrics=False,
            enable_structured_logging=False,
        )
        
        mock_provider = MockEmbeddingProvider()
        chunker = SemanticChunker(config, embedding_provider=mock_provider)
        
        # Chunk single sentence
        chunks = chunker.chunk("This is a single sentence.")
        
        # Should return one chunk
        assert len(chunks) == 1
        
        # Force garbage collection
        gc.collect()
        
        # No embedding storage in strategy after completion
        if chunker._strategy:
            strategy_dict = vars(chunker._strategy)
            for attr_name, attr_value in strategy_dict.items():
                if isinstance(attr_value, np.ndarray):
                    if attr_value.ndim == 2 and attr_value.shape[1] > 100:
                        pytest.fail(
                            f"Strategy retains embeddings in '{attr_name}' "
                            f"after single sentence chunking"
                        )

"""
Property-based tests for batch embedding consolidation using Hypothesis.

These tests verify that batch embedding operations consolidate all texts
into a single API/model call rather than N individual calls, as specified
in requirement 1.1.
"""

from unittest.mock import Mock, patch
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from chunker.embeddings.sentence_transformer import SentenceTransformerProvider
from chunker.embeddings.openai_provider import OpenAIEmbeddingProvider


# Custom strategies for generating test data
@st.composite
def text_list_strategy(draw):
    """Generate a list of non-empty text strings."""
    # Generate 1 to 100 texts
    num_texts = draw(st.integers(min_value=1, max_value=100))
    
    # Generate texts with reasonable content
    texts = []
    for _ in range(num_texts):
        # Generate text with 1-20 words
        num_words = draw(st.integers(min_value=1, max_value=20))
        words = [draw(st.text(alphabet=st.characters(whitelist_categories=('L',)), min_size=1, max_size=10)) 
                 for _ in range(num_words)]
        text = " ".join(words)
        texts.append(text)
    
    return texts


@st.composite
def small_text_list_strategy(draw):
    """Generate a small list of texts (for faster tests)."""
    # Generate 1 to 20 texts
    num_texts = draw(st.integers(min_value=1, max_value=20))
    
    texts = []
    for _ in range(num_texts):
        # Generate simple text
        num_words = draw(st.integers(min_value=1, max_value=10))
        words = [draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=8)) 
                 for _ in range(num_words)]
        text = " ".join(words)
        texts.append(text)
    
    return texts


class TestBatchEmbeddingConsolidationProperties:
    """Property-based tests for batch embedding consolidation."""

    @given(texts=small_text_list_strategy())
    @settings(max_examples=100)
    def test_property_1_sentence_transformer_batch_consolidation(self, texts):
        """
        Feature: chunker-improvements, Property 1: Batch Embedding Consolidation
        
        For any list of texts and embedding model, when batch embedding is enabled,
        all texts should be embedded in a single API/model call rather than N
        individual calls.
        
        **Validates: Requirements 1.1**
        
        This test verifies that SentenceTransformerProvider processes all texts
        in a single model.encode() call.
        """
        # Create provider
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        # Mock the model to track calls
        mock_model = Mock()
        embedding_dim = 384
        mock_model.encode.return_value = np.random.rand(len(texts), embedding_dim)
        provider._model = mock_model
        provider._dimension = embedding_dim
        
        # Embed all texts
        result = provider.embed(texts)
        
        # PROPERTY: The model.encode should be called exactly ONCE for all texts
        assert mock_model.encode.call_count == 1, (
            f"Expected exactly 1 call to model.encode for {len(texts)} texts, "
            f"but got {mock_model.encode.call_count} calls. "
            f"Batch embedding should consolidate all texts into a single call."
        )
        
        # Verify the call was made with all texts
        call_args = mock_model.encode.call_args
        assert call_args is not None, "model.encode should have been called"
        
        # The first positional argument should be the list of texts
        called_with_texts = call_args[0][0]
        assert called_with_texts == texts, (
            f"model.encode should be called with all {len(texts)} texts, "
            f"but was called with {len(called_with_texts)} texts"
        )
        
        # Verify result shape
        assert result.shape == (len(texts), embedding_dim), (
            f"Expected result shape ({len(texts)}, {embedding_dim}), "
            f"got {result.shape}"
        )

    @given(texts=small_text_list_strategy())
    @settings(max_examples=100)
    def test_property_1_openai_batch_consolidation_within_limit(self, texts):
        """
        Feature: chunker-improvements, Property 1: Batch Embedding Consolidation
        
        For any list of texts within the batch size limit, OpenAI provider should
        consolidate all texts into a single API call.
        
        **Validates: Requirements 1.1**
        """
        # Ensure we're within batch limit for this test
        if len(texts) > 100:
            texts = texts[:100]
        
        # Create provider with large batch size
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=2047  # OpenAI's max limit
        )
        
        # Mock the client
        mock_client = Mock()
        embedding_dim = 1536
        
        # Create mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * embedding_dim) for _ in range(len(texts))
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client
        
        # Embed all texts
        result = provider.embed(texts)
        
        # PROPERTY: The API should be called exactly ONCE for all texts within batch limit
        assert mock_client.embeddings.create.call_count == 1, (
            f"Expected exactly 1 API call for {len(texts)} texts (within batch limit), "
            f"but got {mock_client.embeddings.create.call_count} calls. "
            f"Batch embedding should consolidate all texts into a single API call."
        )
        
        # Verify the call was made with all texts
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert 'input' in call_kwargs, "API call should include 'input' parameter"
        
        called_with_texts = call_kwargs['input']
        assert len(called_with_texts) == len(texts), (
            f"API call should include all {len(texts)} texts, "
            f"but was called with {len(called_with_texts)} texts"
        )
        
        # Verify result shape
        assert result.shape == (len(texts), embedding_dim), (
            f"Expected result shape ({len(texts)}, {embedding_dim}), "
            f"got {result.shape}"
        )

    @given(
        num_texts=st.integers(min_value=1, max_value=50),
        batch_size=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_1_batch_consolidation_minimizes_calls(self, num_texts, batch_size):
        """
        Feature: chunker-improvements, Property 1: Batch Embedding Consolidation
        
        For any list of N texts with batch size B, the number of API calls should
        be ceil(N/B), which is the minimum possible number of calls.
        
        **Validates: Requirements 1.1**
        """
        # Generate simple texts
        texts = [f"text {i}" for i in range(num_texts)]
        
        # Create provider with specified batch size
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=batch_size
        )
        
        # Mock the client
        mock_client = Mock()
        embedding_dim = 1536
        
        def create_response(model, input):
            """Create mock response matching input size."""
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * embedding_dim) for _ in range(len(input))
            ]
            return mock_response
        
        mock_client.embeddings.create.side_effect = create_response
        provider._client = mock_client
        
        # Embed all texts
        result = provider.embed(texts)
        
        # Calculate expected number of calls
        import math
        expected_calls = math.ceil(num_texts / batch_size)
        
        # PROPERTY: Number of API calls should be ceil(N/B) - the minimum possible
        actual_calls = mock_client.embeddings.create.call_count
        assert actual_calls == expected_calls, (
            f"For {num_texts} texts with batch_size={batch_size}, "
            f"expected {expected_calls} API calls (ceil({num_texts}/{batch_size})), "
            f"but got {actual_calls} calls. "
            f"Batch embedding should minimize the number of API calls."
        )
        
        # Verify result shape
        assert result.shape == (num_texts, embedding_dim), (
            f"Expected result shape ({num_texts}, {embedding_dim}), "
            f"got {result.shape}"
        )

    @given(texts=st.lists(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=8),
        min_size=2,  # At least 2 texts to test batch consolidation
        max_size=20
    ))
    @settings(max_examples=50)
    def test_property_1_no_individual_calls_for_batch(self, texts):
        """
        Feature: chunker-improvements, Property 1: Batch Embedding Consolidation
        
        For any list of texts, batch embedding should NOT make individual calls
        for each text. Instead, it should consolidate them.
        
        **Validates: Requirements 1.1**
        """
        # Create provider
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        # Mock the model
        mock_model = Mock()
        embedding_dim = 384
        mock_model.encode.return_value = np.random.rand(len(texts), embedding_dim)
        provider._model = mock_model
        provider._dimension = embedding_dim
        
        # Embed all texts
        result = provider.embed(texts)
        
        # PROPERTY: Should NOT make N individual calls (one per text)
        # Should make significantly fewer calls (ideally 1)
        actual_calls = mock_model.encode.call_count
        num_texts = len(texts)
        
        assert actual_calls < num_texts, (
            f"Batch embedding made {actual_calls} calls for {num_texts} texts. "
            f"This suggests individual calls rather than batch consolidation. "
            f"Expected significantly fewer calls (ideally 1)."
        )
        
        # Verify result shape
        assert result.shape == (num_texts, embedding_dim)


class TestBatchEmbeddingConsolidationEdgeCases:
    """Unit tests for edge cases in batch embedding consolidation."""
    
    def test_single_text_still_uses_batch_interface(self):
        """Single text should still use the batch interface (list input)."""
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 384)
        provider._model = mock_model
        provider._dimension = 384
        
        texts = ["single text"]
        result = provider.embed(texts)
        
        # Should make exactly 1 call
        assert mock_model.encode.call_count == 1
        
        # Should be called with a list (batch interface)
        call_args = mock_model.encode.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args == texts
        
        # Result should be 2D array
        assert result.shape == (1, 384)
    
    def test_empty_list_makes_no_calls(self):
        """Empty list should not make any API/model calls."""
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        mock_model = Mock()
        provider._model = mock_model
        provider._dimension = 384
        
        result = provider.embed([])
        
        # Should make NO calls for empty list
        assert mock_model.encode.call_count == 0
        
        # Should return empty array with correct shape
        assert result.shape == (0, 384)
    
    def test_large_batch_consolidation(self):
        """Large batch should still consolidate into minimal calls."""
        # Test with 1000 texts
        texts = [f"text {i}" for i in range(1000)]
        
        provider = SentenceTransformerProvider(
            model_name="all-MiniLM-L6-v2",
            batch_size=64  # Internal batch size for memory efficiency
        )
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1000, 384)
        provider._model = mock_model
        provider._dimension = 384
        
        result = provider.embed(texts)
        
        # Should make exactly 1 call (model handles internal batching)
        assert mock_model.encode.call_count == 1
        
        # Should be called with all 1000 texts
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args) == 1000
        
        # Result should have correct shape
        assert result.shape == (1000, 384)
    
    def test_openai_respects_batch_limit(self):
        """OpenAI provider should respect batch size limit but minimize calls."""
        # Test with 250 texts and batch_size=100
        texts = [f"text {i}" for i in range(250)]
        
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=100
        )
        
        mock_client = Mock()
        
        def create_response(model, input):
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(len(input))]
            return mock_response
        
        mock_client.embeddings.create.side_effect = create_response
        provider._client = mock_client
        
        result = provider.embed(texts)
        
        # Should make exactly 3 calls (ceil(250/100) = 3)
        assert mock_client.embeddings.create.call_count == 3
        
        # Verify batch sizes: 100, 100, 50
        call_sizes = []
        for call in mock_client.embeddings.create.call_args_list:
            call_sizes.append(len(call[1]['input']))
        
        assert call_sizes == [100, 100, 50], (
            f"Expected batch sizes [100, 100, 50], got {call_sizes}"
        )
        
        # Result should have correct shape
        assert result.shape == (250, 1536)

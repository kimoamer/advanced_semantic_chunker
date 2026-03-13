"""
Tests for batch embedding support in embedding providers.

This module tests that embedding providers properly support batch operations
with single API/model calls as specified in requirement 1.1.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from chunker.embeddings.sentence_transformer import SentenceTransformerProvider
from chunker.embeddings.openai_provider import OpenAIEmbeddingProvider


class TestSentenceTransformerBatchEmbedding:
    """Test batch embedding support for SentenceTransformerProvider."""

    def test_embed_accepts_list_of_texts(self):
        """Test that embed() method accepts a list of texts."""
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        # Mock the model to avoid loading
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 384)
        provider._model = mock_model
        provider._dimension = 384
        
        texts = ["text 1", "text 2", "text 3"]
        result = provider.embed(texts)
        
        # Verify the model.encode was called once with all texts
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == texts
        
        # Verify result shape
        assert result.shape == (3, 384)

    def test_embed_processes_all_texts_in_single_call(self):
        """Test that all texts are processed in a single model call."""
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        # Mock the model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(10, 384)
        provider._model = mock_model
        provider._dimension = 384
        
        texts = [f"text {i}" for i in range(10)]
        provider.embed(texts)
        
        # Verify encode was called exactly once (single call for all texts)
        assert mock_model.encode.call_count == 1

    def test_embed_empty_list_returns_empty_array(self):
        """Test that embedding an empty list returns an empty array."""
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
        
        # Mock the model
        mock_model = Mock()
        provider._model = mock_model
        provider._dimension = 384
        
        result = provider.embed([])
        
        # Verify no model call was made
        mock_model.encode.assert_not_called()
        
        # Verify empty array with correct shape
        assert result.shape == (0, 384)


class TestOpenAIBatchEmbedding:
    """Test batch embedding support for OpenAIEmbeddingProvider."""

    def test_embed_accepts_list_of_texts(self):
        """Test that embed() method accepts a list of texts."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key"
        )
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
            Mock(embedding=[0.3] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client
        
        texts = ["text 1", "text 2", "text 3"]
        result = provider.embed(texts)
        
        # Verify the API was called
        mock_client.embeddings.create.assert_called_once()
        
        # Verify result shape
        assert result.shape == (3, 1536)

    def test_embed_single_api_call_within_batch_limit(self):
        """Test that texts within batch_size are sent in a single API call."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=100
        )
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        # Create 50 mock embeddings
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(50)]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client
        
        texts = [f"text {i}" for i in range(50)]
        result = provider.embed(texts)
        
        # Verify only one API call was made (single call for batch)
        assert mock_client.embeddings.create.call_count == 1
        
        # Verify all texts were sent in the call
        call_args = mock_client.embeddings.create.call_args
        assert len(call_args[1]['input']) == 50

    def test_embed_multiple_api_calls_exceeding_batch_limit(self):
        """Test that texts exceeding batch_size are split into multiple calls."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=100
        )
        
        # Mock the client
        mock_client = Mock()
        
        def create_response(model, input):
            # Return mock response with correct number of embeddings
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(len(input))]
            return mock_response
        
        mock_client.embeddings.create.side_effect = create_response
        provider._client = mock_client
        
        # Create 150 texts (should require 2 API calls with batch_size=100)
        texts = [f"text {i}" for i in range(150)]
        result = provider.embed(texts)
        
        # Verify two API calls were made
        assert mock_client.embeddings.create.call_count == 2
        
        # Verify result shape
        assert result.shape == (150, 1536)

    def test_default_batch_size_is_max_limit(self):
        """Test that default batch_size is set to OpenAI's max limit."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key"
        )
        
        # Verify default batch_size is 2047 (OpenAI's limit)
        assert provider.batch_size == 2047

    def test_batch_size_capped_at_max_limit(self):
        """Test that batch_size is capped at OpenAI's max limit."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key",
            batch_size=5000  # Exceeds limit
        )
        
        # Verify batch_size is capped at 2047
        assert provider.batch_size == 2047

    def test_embed_empty_list_returns_empty_array(self):
        """Test that embedding an empty list returns an empty array."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key"
        )
        
        # Mock the client
        mock_client = Mock()
        provider._client = mock_client
        
        result = provider.embed([])
        
        # Verify no API call was made
        mock_client.embeddings.create.assert_not_called()
        
        # Verify empty array with correct shape
        assert result.shape == (0, 1536)

    def test_embed_handles_empty_strings(self):
        """Test that empty strings are replaced with spaces."""
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            api_key="test-key"
        )
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client
        
        texts = ["text 1", ""]
        provider.embed(texts)
        
        # Verify empty string was replaced with space
        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]['input'] == ["text 1", " "]
